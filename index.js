const express = require("express");
const OpenAI = require("openai");
const axios = require("axios");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");
const { OpenAIEmbeddings } = require("@langchain/openai");
const { convert } = require("html-to-text");

require("dotenv").config();

const app = express();
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const port = 3080;

app.post("/search", async (req, res) => {
  try {
    const { query } = req.body;
    const result = await search(query);
    res.status(200).json({ message: "Successful!", ...result });
  } catch (error) {
    res.status(500).json({ message: "Something went wrong, please try again" });
    console.error("Error occurred", error);
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

async function getSearchResults(optimizedQuery) {
  const response = await axios.get(
    "https://api.search.brave.com/res/v1/web/search",
    {
      headers: {
        Accept: "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": process.env.BRAVE_API_KEY,
      },
      params: {
        q: optimizedQuery,
        count: 5,
      },
    }
  );

  // return top 5
  return response.data.web?.results || [];
}

async function getPageContent(link) {
  const response = await axios.get(link, { timeout: 5000 });
  const text = response.data;
  return convert(text, { wordwrap: 130 });
}

function cleanText(text) {
  return text.trim().replace(/\s+|\n+/g, " "); // Collapses all spaces and newlines into a single space
}

async function extractContextFromResultsWithRAG(query, searchResults) {
  const getRelevantContext = async (query, source) => {
    const content = await getPageContent(source.url);
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2000,
      chunkOverlap: 10,
    });

    const chunks = await splitter.splitText(content);

    const vectorStore = await MemoryVectorStore.fromTexts(
      chunks,
      {},
      new OpenAIEmbeddings()
    );

    const result = await vectorStore.similaritySearch(query, 1);

    return `source:${source.url} content: ${result[0]?.pageContent}`;
  };

  const promises = searchResults.map((source) =>
    getRelevantContext(query, source)
  );

  const result = await Promise.allSettled(promises);

  return result
    .filter((res) => res.status === "fulfilled")
    .map((res) => `${cleanText(res.value)}---\n`)
    .join("");
}

async function extractContextFromResultsWithoutRAG(searchResults) {
  const getRelevantContext = async (source) => {
    const content = await getPageContent(source.url);
    return content;
  };

  const promises = searchResults.map((source) => getRelevantContext(source));

  const result = await Promise.allSettled(promises);

  return result
    .filter((res) => res.status === "fulfilled")
    .map((res) => `${res.value.trim()}---\n`)
    .join("");
}

async function getRelatedQueries(query) {
  const messages = [
    {
      role: "system",
      content: `You are a search engine expert tasked with getting related search queries. Output as a JSON array of strings (related_search_queries)`,
    },
    {
      role: "user",
      content: `User's Query: "${query}" Get 5 related search queries:`,
    },
  ];

  const completion = await openai.chat.completions.create({
    messages,
    model: "gpt-3.5-turbo",
    temperature: 0.8,
    response_format: { type: "json_object" },
  });

  return (
    JSON.parse(completion.choices[0].message.content)?.related_search_queries ||
    []
  );
}

async function getAnswer(query, context) {
  const messages = [
    {
      role: "system",
      content: `You are an AI-chatbot-powered research and conversational search engine that answers user search queries`,
    },
    {
      role: "user",
      content: `Context sections: "${context}" Question: "${query}" Answer in detail:`,
    },
  ];

  const completion = await openai.chat.completions.create({
    messages,
    model: "gpt-4o",
    temperature: 0.5,
  });

  return completion.choices[0].message.content;
}

async function search(query) {
  const searchResults = await getSearchResults(query); // Get Search Results Using Brave Search API
  const context = await extractContextFromResultsWithRAG(query, searchResults);

  // Running concurrently
  const [answer, relatedQueries] = await Promise.all([
    getAnswer(query, context),
    getRelatedQueries(query),
  ]);

  return {
    answer,
    relatedQueries,
    sources: searchResults.map((source) => ({
      title: source.title,
      url: source.url,
      icon: source.profile.img,
      name: source.profile.name,
    })),
  };
}


