import functools
import math
import os  # noqa: F401
from random import choice, randint
from time import time

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm

import faiss  # noqa: F401
import nlp  # noqa: F401
import pandas as pd
from elasticsearch import Elasticsearch  # noqa: F401
from elasticsearch.helpers import bulk, streaming_bulk  # noqa: F401
from transformers import AdamW, AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


pd.set_option("display.max_colwidth", None)

###############
# ELI5 seq2seq model training
###############
class ELI5DatasetS2S(Dataset):
    def __init__(
        self, examples_array, make_doc_fun=None, extra_answer_threshold=3, document_cache=None, training=True
    ):
        self.training = training
        self.data = examples_array
        self.make_doc_function = make_doc_fun
        self.document_cache = {} if document_cache is None else document_cache
        assert not (make_doc_fun is None and document_cache is None)
        # make index of specific question-answer pairs from multi-answers
        if self.training:
            self.qa_id_list = [
                (i, j)
                for i, qa in enumerate(self.data)
                for j, (a, sc) in enumerate(zip(qa["answers"]["text"], qa["answers"]["score"]))
                if j == 0 or sc >= extra_answer_threshold
            ]
        else:
            self.qa_id_list = [(i, 0) for i in range(self.data.num_rows)]

    def __len__(self):
        return len(self.qa_id_list)

    def make_example(self, idx):
        i, j = self.qa_id_list[idx]
        example = self.data[i]
        question = example["title"] + " " + example["selftext"]
        answer = example["answers"]["text"][j]
        q_id = example["q_id"]
        if self.make_doc_function is not None:
            self.document_cache[q_id] = self.document_cache.get(q_id, self.make_doc_function(example["title"]))
        document = self.document_cache[q_id]
        in_st = "question: {} context: {}".format(
            question.lower().replace(" --t--", "").strip(), document.lower().strip(),
        )
        out_st = answer
        return (in_st, out_st)

    def __getitem__(self, idx):
        return self.make_example(idx)


def make_qa_s2s_model(model_name="facebook/bart-large", from_file=None, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if from_file is not None:
        param_dict = torch.load(from_file)  # has model weights, optimizer, and scheduler states
        model.load_state_dict(param_dict["model"])
    return tokenizer, model


def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=360, device="cuda:0"):
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=min(max_len, max_a_len), pad_to_max_length=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks["input_ids"]).to(device),
        torch.LongTensor(a_toks["attention_mask"]).to(device),
    )
    lm_labels = a_ids[:, 1:].contiguous().clone()
    lm_labels[a_mask[:, 1:].contiguous() == 0] = -100
    model_inputs = {
        "input_ids": q_ids,
        "attention_mask": q_mask,
        "decoder_input_ids": a_ids[:, :-1].contiguous(),
        "lm_labels": lm_labels,
    }
    return model_inputs


def train_qa_s2s_epoch(model, dataset, tokenizer, optimizer, scheduler, args, e=0, curriculum=False):
    model.train()
    # make iterator
    if curriculum:
        train_sampler = SequentialSampler(dataset)
    else:
        train_sampler = RandomSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device="cuda:0"
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    for step, batch_inputs in enumerate(epoch_iterator):
        pre_loss = model(**batch_inputs)[0]
        loss = pre_loss.sum() / pre_loss.shape[0]
        loss.backward()
        # optimizer
        if step % args.backward_freq == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        # some printing within the epoch
        loc_loss += loss.item()
        loc_steps += 1
        if step % args.print_freq == 0 or step == 1:
            print(
                "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                    e, step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time,
                )
            )
            loc_loss = 0
            loc_steps = 0


def eval_qa_s2s_epoch(model, dataset, tokenizer, args):
    model.eval()
    # make iterator
    train_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device="cuda:0"
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    with torch.no_grad():
        for step, batch_inputs in enumerate(epoch_iterator):
            pre_loss = model(**batch_inputs)[0]
            loss = pre_loss.sum() / pre_loss.shape[0]
            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0:
                print(
                    "{:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                        step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time,
                    )
                )
    print("Total \t L: {:.3f} \t -- {:.3f}".format(loc_loss / loc_steps, time() - st_time,))


def train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer, s2s_train_dset, s2s_valid_dset, s2s_args):
    s2s_optimizer = AdamW(qa_s2s_model.parameters(), lr=s2s_args.learning_rate, eps=1e-8)
    s2s_scheduler = get_linear_schedule_with_warmup(
        s2s_optimizer,
        num_warmup_steps=400,
        num_training_steps=(s2s_args.num_epochs + 1) * math.ceil(len(s2s_train_dset) / s2s_args.batch_size),
    )
    for e in range(s2s_args.num_epochs):
        train_qa_s2s_epoch(
            qa_s2s_model,
            s2s_train_dset,
            qa_s2s_tokenizer,
            s2s_optimizer,
            s2s_scheduler,
            s2s_args,
            e,
            curriculum=(e == 0),
        )
        m_save_dict = {
            "model": qa_s2s_model.state_dict(),
            "optimizer": s2s_optimizer.state_dict(),
            "scheduler": s2s_scheduler.state_dict(),
        }
        print("Saving model {}".format(s2s_args.model_save_name))
        eval_qa_s2s_epoch(qa_s2s_model, s2s_valid_dset, qa_s2s_tokenizer, s2s_args)
        torch.save(m_save_dict, "{}_{}.pth".format(s2s_args.model_save_name, e))


# generate answer from input "question: ... context: <p> ..."
def qa_s2s_generate(
    question_doc,
    qa_s2s_model,
    qa_s2s_tokenizer,
    num_answers=1,
    num_beams=None,
    min_len=64,
    max_len=256,
    do_sample=False,
    temp=1.0,
    top_p=None,
    top_k=None,
    max_input_length=512,
    device="cuda:0",
):
    model_inputs = make_qa_s2s_batch([(question_doc, "A")], qa_s2s_tokenizer, max_input_length, device=device,)
    n_beams = num_answers if num_beams is None else max(num_beams, num_answers)
    generated_ids = qa_s2s_model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        min_length=min_len,
        max_length=max_len,
        do_sample=do_sample,
        early_stopping=True,
        num_beams=1 if do_sample else n_beams,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=qa_s2s_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=num_answers,
        decoder_start_token_id=qa_s2s_tokenizer.bos_token_id,
    )
    return [qa_s2s_tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]


###############
# ELI5-trained retrieval model usage
###############
def embed_passages_for_retrieval(passages, tokenizer, qa_embedder, max_length=128, device="cuda:0"):
    a_toks = tokenizer.batch_encode_plus(passages, max_length=max_length, pad_to_max_length=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks["input_ids"]).to(device),
        torch.LongTensor(a_toks["attention_mask"]).to(device),
    )
    with torch.no_grad():
        a_reps = qa_embedder.embed_answers(a_ids, a_mask).cpu().type(torch.float)
    return a_reps.numpy()


def embed_questions_for_retrieval(q_ls, tokenizer, qa_embedder, device="cuda:0"):
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=128, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    with torch.no_grad():
        q_reps = qa_embedder.embed_questions(q_ids, q_mask).cpu().type(torch.float)
    return q_reps.numpy()


def make_qa_dense_index(
    qa_embedder,
    tokenizer,
    passages_dset,
    batch_size=512,
    max_length=128,
    index_name="kilt_passages_reps.dat",
    dtype="float32",
    device="cuda:0",
):
    st_time = time()
    fp = np.memmap(index_name, dtype=dtype, mode="w+", shape=(passages_dset.num_rows, 128))
    n_batches = math.ceil(passages_dset.num_rows / batch_size)
    for i in range(n_batches):
        passages = [p for p in passages_dset[i * batch_size : (i + 1) * batch_size]["passage_text"]]
        reps = embed_passages_for_retrieval(passages, tokenizer, qa_embedder, max_length, device)
        fp[i * batch_size : (i + 1) * batch_size] = reps
        if i % 50 == 0:
            print(i, time() - st_time)


def evaluate_retriever(qa_list, retriever_func, scoring_func, n_ret=10, verbose=False):
    total_retriever_time = 0.0
    total_retriever_score = 0.0
    st_time = time()
    for i, (question, answer) in enumerate(qa_list):
        r_time = time()
        retrieved_passages = retriever_func(question, n_ret)
        total_retriever_time += time() - r_time
        total_retriever_score += scoring_func(retrieved_passages, answer)
        if verbose and ((i + 1) % 500 == 0 or i <= 1):
            print(
                "{:03d}: S-{:.4f} T-{:.4f} | {:.2f}".format(
                    i + 1, total_retriever_score / (i + 1), total_retriever_time / (i + 1), time() - st_time
                )
            )
    return {"idf_recall": total_retriever_score / (i + 1), "retrieval_time": total_retriever_time / (i + 1)}


# build a support document for the question out of Wikipedia snippets
def query_qa_dense_index(
    question, qa_embedder, tokenizer, wiki_passages, wiki_index, n_results=10, min_length=20, device="cuda:0"
):
    q_rep = embed_questions_for_retrieval([question], tokenizer, qa_embedder, device=device)
    D, I = wiki_index.search(q_rep, 2 * n_results)
    res_passages = [wiki_passages[int(i)] for i in I[0]]
    support_doc = "<P> " + " <P> ".join([p["passage_text"] for p in res_passages])
    res_list = [dict([(k, p[k]) for k in wiki_passages.column_names]) for p in res_passages]
    res_list = [res for res in res_list if len(res["passage_text"].split()) > min_length][:n_results]
    for r, sc in zip(res_list, D[0]):
        r["score"] = float(sc)
    return support_doc, res_list


def batch_query_qa_dense_index(questions, qa_embedder, tokenizer, wiki_passages, wiki_index, n_results=10):
    q_rep = embed_questions_for_retrieval(questions, tokenizer, qa_embedder)
    D, I = wiki_index.search(q_rep, n_results)
    res_passages_lst = [[wiki_passages[int(i)] for i in i_lst] for i_lst in I]
    support_doc_lst = [
        "<P> " + " <P> ".join([p["passage_text"] for p in res_passages]) for res_passages in res_passages_lst
    ]
    all_res_lists = []
    for (res_passages, dl) in zip(res_passages_lst, D):
        res_list = [dict([(k, p[k]) for k in wiki_passages.column_names]) for p in res_passages]
        for r, sc in zip(res_list, dl):
            r["score"] = float(sc)
        all_res_lists += [res_list[:]]
    return support_doc_lst, all_res_lists


# find nearest neighbors of an answer or declarative text in Wikipedia snippets
def query_qa_dense_index_nn(passage, qa_embedder, tokenizer, wiki_passages, wiki_index, n_results=10, min_length=20):
    a_rep = embed_passages_for_retrieval([passage], tokenizer, qa_embedder)
    D, I = wiki_index.search(a_rep, 2 * n_results)
    res_passages = [wiki_passages[int(i)] for i in I[0]]
    support_doc = "<P> " + " <P> ".join([p["passage_text"] for p in res_passages])
    res_list = [dict([(k, p[k]) for k in wiki_passages.column_names]) for p in res_passages]
    res_list = [res for res in res_list if len(res["passage_text"].split()) > min_length][:n_results]
    for r, sc, i in zip(res_list, D[0], I[0]):
        r["passage_id"] = int(i)
        r["score"] = float(sc)
    return support_doc, res_list


def batch_query_qa_dense_index_nn(passages, qa_embedder, tokenizer, wiki_passages, wiki_index, n_results=10):
    a_reps = embed_passages_for_retrieval(passages, tokenizer, qa_embedder)
    D, I = wiki_index.search(a_reps, n_results)
    res_passages_lst = [[wiki_passages[int(i)] for i in i_lst] for i_lst in I]
    support_doc_lst = [
        "<P> " + " <P> ".join([p["passage_text"] for p in res_passages]) for res_passages in res_passages_lst
    ]
    all_res_lists = []
    for (res_passages, dl, il) in zip(res_passages_lst, D, I):
        res_list = [dict([(k, p[k]) for k in wiki_passages.column_names]) for p in res_passages]
        for r, sc, i in zip(res_list, dl, il):
            r["passage_id"] = int(i)
            r["score"] = float(sc)
        all_res_lists += [res_list[:]]
    return support_doc_lst, all_res_lists