import { ChatOpenAI } from "langchain/chat_models/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PromptTemplate } from "langchain/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "langchain/schema/runnable";
import { StringOutputParser } from "langchain/schema/output_parser";
import { Document } from "langchain/document";

// 创建llm实例，需要传入key
const model = new ChatOpenAI({});

// 创建 词嵌入embedding 实例，词向量
const vectorStore = await HNSWLib.fromTexts(
  ["mitochondria is the powerhouse of the cell"],
  [{ id: 1 }],
  new OpenAIEmbeddings()
);

// 获取 Document retriever，文档/向量 查找器
const retriever = vectorStore.asRetriever();

// prompt模板
const prompt =
  PromptTemplate.fromTemplate(`Answer the question based only on the following context:
{context}

Question: {question}`);

// 文档序列化
const serializeDocs = (docs: Document[]) =>
  docs.map((doc) => doc.pageContent).join("\n");

// 创建 运行队列，RunnableSequence.from，获取 chain实例
const chain = RunnableSequence.from([
  {
    context: retriever.pipe(serializeDocs),
    question: new RunnablePassthrough(),
  },
  prompt,
  model,
  new StringOutputParser(),
]);

const result = await chain.invoke("What is the powerhouse of the cell?");

console.log(result);

/*
  "The powerhouse of the cell is the mitochondria."
*/