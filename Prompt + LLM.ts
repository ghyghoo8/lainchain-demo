/**
 * @step: PromptTemplate / ChatPromptTemplate -> LLM / ChatModel -> OutputParser
 */

import { PromptTemplate } from 'langchain/prompts'
import { ChatOpenAI } from "langchain/chat_models/openai";

// 创建llm 实例
const model = new ChatOpenAI({});

// 创建prompt模板实例
const promptTemp = PromptTemplate.fromTemplate("Tell me a joke about {topic}")

// 获取 langchain实例
const chain = promptTemp.pipe(model)

const result = chain.invoke({
  topic: '饺子🥟'
})

// 默认格式化输出
/*
  AIMessage {
    content: "Why don't bears wear shoes?\n\nBecause they have bear feet!",
  }
*/
console.log('result===>', result)