import express from "express"
import dotenv from "dotenv"
import connectDb from "./lib/db.js"
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"
import { ChatGroq } from "@langchain/groq"
import { Annotation, StateGraph } from "@langchain/langgraph"

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

const llm = new ChatGroq({
    model: "openai/gpt-oss-120b",
    apiKey: process.env.GROQ_API_KEY,
    temperature: 0.7
});




const State = Annotation.Root({
    prompt: Annotation,
    aiMsg: Annotation
})

const callLLM = async (state) => {
    console.log("state:", state)
    const response = await llm.invoke([
        {
            role: "system",
            content: "You are a helpful assistant. Your name is jarvis. If you don't know the answer, say sorry, I am unable to provide or whatever suitable",
        },
        { role: "human", content: state.prompt },
    ]);
    return { aiMsg: response.content }
}

const graph = new StateGraph(State)
    .addNode("agent", callLLM)
    .addEdge("__start__", "agent")
    .addEdge("agent", "__end__")
    .compile()

app.post("/ai", async (req, res) => {
    const { input } = req.body;

    const response = await graph.invoke({ prompt: input })
    console.log(response)

    return res.status(200).json({ "ai": response });
});
/**
 * Date : 30.09.2026
 * Topic : LangGraph
 * Why we need langGraph
 * LangGraph Workflow
 * Graph,Edge,Condtional Edges,Node,Tool Node,Web Search Tool, State, Custom state
 * 
 * */
