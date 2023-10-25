import { DataSource } from "typeorm";
import { SqlDatabase } from "langchain/sql_db";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "langchain/schema/runnable";
import { PromptTemplate } from "langchain/prompts";
import { StringOutputParser } from "langchain/schema/output_parser";
import { ChatOpenAI } from "langchain/chat_models/openai";

// 链接sqlite数据库
const datasource = new DataSource({
  type: "sqlite",
  database: "Chinook.db",
});

// 创建工作流 db实例
const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: datasource,
});

// prompt模板，定义fields
const prompt =
  PromptTemplate.fromTemplate(`Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:`);

// llm实例
const model = new ChatOpenAI();

// The `RunnablePassthrough.assign()` is used here to passthrough the input from the `.invoke()`
// call (in this example it's the question), along with any inputs passed to the `.assign()` method.
// In this case, we're passing the schema.
const sqlQueryGeneratorChain = RunnableSequence.from([
  // PromptTemplate fields
  RunnablePassthrough.assign({
    schema: async () => db.getTableInfo(),
  }),
  // PromptTemplate
  prompt,
  // llm
  model.bind({ stop: ["\nSQLResult:"] }),
  // format
  new StringOutputParser(),
]);

const result = await sqlQueryGeneratorChain.invoke({
  question: "How many employees are there?",
});

console.log({
  result,
});

/*
  {
    result: "SELECT COUNT(EmployeeId) AS TotalEmployees FROM Employee"
  }
*/

const finalResponsePrompt =
  PromptTemplate.fromTemplate(`Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}`);

/**
 * 步骤：
 * 1. 描述任务
 * 2. 提供 dbTable db.getTableInfo，schema
 * 3. 生成sql返回===>db.run(input.query)
 * 4. 对查询结果 以NL方式 做描述 / 或者可以做其他格式的描述
 */
const fullChain = RunnableSequence.from([
  RunnablePassthrough.assign({
    query: sqlQueryGeneratorChain,
  }),
  {
    schema: async () => db.getTableInfo(),
    question: (input) => input.question,
    query: (input) => input.query,
    response: (input) => db.run(input.query),
  },
  finalResponsePrompt,
  model,
]);

const finalResponse = await fullChain.invoke({
  question: "How many employees are there?",
});

console.log(finalResponse);

/*
  AIMessage {
    content: 'There are 8 employees.',
    additional_kwargs: { function_call: undefined }
  }
*/