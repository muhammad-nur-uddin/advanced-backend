import express from "express"
import dotenv from "dotenv"
import connectDb from "./lib/db.js"
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"
import { ChatGroq } from "@langchain/groq"
import { Annotation, MemorySaver, MessagesAnnotation, StateGraph } from "@langchain/langgraph"
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";
import fs from "fs"
import { PDFParse } from "pdf-parse"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { QdrantVectorStore } from "@langchain/qdrant"

dotenv.config()

const port = process.env.port

const app = express()
app.use(express.json());

app.get("/", (req, res) => {
    return res.status(200).json({ message: "hello from redis!" })
})

app.listen(port, () => {
    // connectDb()
    console.log(`server is runnig at port ${port}`)
})

const storage = new MemorySaver()

const tool = new TavilySearch({
    maxResults: 5,
    topic: "general",
});

const tools = [tool]

const toolNode = new ToolNode(tools)

const llm = new ChatGroq({
    model: "openai/gpt-oss-120b",
    apiKey: process.env.GROQ_API_KEY,
    temperature: 0.7,
    // maxTokens: 100,
    maxRetries: 2
}).bindTools(tools);




const callLLM = async (state) => {
    console.log("state:", state)
    const response = await llm.invoke([
        {
            role: "system",
            content: `You are Jarvis AI assistant

Use conversation memory first.

Only use tools when the answer requires
external real-time information like:
weather, news, web search, stock prices etc.

Do NOT call tools for simple conversation,
memory-based questions, greetings,
or personal context`
        },
        ...state.messages,
    ]);
    return { messages: [response] }
}

const shouldContinue = async (state) => {
    const lastMessage = state.messages[state.messages.length - 1]
    if (lastMessage.tool_calls.length > 0) {
        return "tools"
    } else {
        return "__end__"
    }
}

const graph = new StateGraph(MessagesAnnotation)
    .addNode("agent", callLLM)
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addEdge("tools", "agent")
    .addConditionalEdges("agent", shouldContinue)
    .compile({ checkpointer: storage })

const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "gemini-embedding-001", // 768 dimensions
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    title: "Document title",
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
    collectionName: "my-grocery-store",
});

const upload = async () => {
    const pdfPath = "./knowledge.pdf"
    const buffer = fs.readFileSync(pdfPath)
    const pdfResult = await (await new PDFParse({ data: buffer }).getText()).text

    const docs = await new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    }).createDocuments([pdfResult])
    console.log(docs)
    
    await vectorStore.addDocuments(docs)
}

upload()



app.post("/ai", async (req, res) => {
    const { input } = req.body;

    const response = await graph.invoke({
        messages: [{
            role: "human",
            content: input
        }]
    }, { configurable: { thread_id: "user123" } })
    console.log(response.messages[response.messages.length - 1].content)

    return res.status(200).json({ "ai": response.messages[response.messages.length - 1].content });
});

/**
 * Date : 30.09.2026
 * Topic : LangGraph
 * Why we need langGraph
 * LangGraph Workflow
 * Graph,Edge,Condtional Edges,Node,Tool Node,Web Search Tool, State, Custom state
 * 
 * */
