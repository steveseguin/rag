// rag.js
const EMBEDDING_MODEL = "granite-embedding:30m"; // or your preferred model
const COMPLETION_MODEL = "llama3.2:latest";
const OLLAMA_ENDPOINT = "http://localhost:11434";
const EMBEDDINGS_STORE_NAME = 'embeddings';
const CHUNK_SIZE = 512;
const OVERLAP_SIZE = 50;
const MAX_RECURSIVE_SEARCHES = 10;
const TARGET_TOKEN_COUNT = 8000;


const initialContext = "VDO.Ninja is a free, open-source web service that allows you to bring live video and audio from a smartphone, tablet, or remote computer directly into video production software such as OBS Studio, vMix, or Streamlabs. \nVDO.Ninja works by using WebRTC, a peer-to-peer streaming technology that is built into most modern web browsers. This allows for very low-latency and high-quality video and audio streaming, even across the Internet. \nVDO.Ninja supports a wide range of use cases, including:\nTurning a mobile device into a wireless webcam \nStreaming high-quality audio and video across the Internet or within a LAN \nRecording remote or local video without needing any downloads \nApplying green screens, digital face effects, and other advanced video filters to video streams \nAnd more! \nVDO.Ninja is also compatible with a wide range of devices, including smartphones, tablets, laptops, desktops, and even Raspberry Pis. \nHere are some of the key concepts and terms used in VDO.Ninja:\nView Link: A URL that is used to view a video stream. \nPush Link: A URL that is used to publish a video stream. \nRoom: A virtual room where multiple devices can connect to share audio and video. \nDirector: A user who has control over a room. \nGuest: A user who joins a room. \nScene: A collection of video streams that are mixed together. \nURL Parameter: A query string parameter that can be added to a VDO.Ninja URL to customize the behavior of the service.";

// IndexedDB setup for storing embeddings
async function openEmbeddingsDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('EmbeddingsDB', 1);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains(EMBEDDINGS_STORE_NAME)) {
                db.createObjectStore(EMBEDDINGS_STORE_NAME, { keyPath: 'id' });
            }
        };
    });
}

async function searchRAGWithTokens(query, targetTokens = TARGET_TOKEN_COUNT) {
    const queryEmbedding = await getEmbedding(query);
    const db = await openEmbeddingsDB();
    const transaction = db.transaction(EMBEDDINGS_STORE_NAME, 'readonly');
    const store = transaction.objectStore(EMBEDDINGS_STORE_NAME);
    
    return new Promise((resolve, reject) => {
        const chunks = [];
        let totalTokens = 0;
        
        store.openCursor().onsuccess = (event) => {
            const cursor = event.target.result;
            if (cursor) {
                const chunk = cursor.value;
                const semanticScore = cosineSimilarity(queryEmbedding, chunk.embedding);
                const contextScore = chunk.metadata.precedingContext && chunk.metadata.followingContext ? 0.2 : 0;
                const finalScore = semanticScore + contextScore;
                
                chunks.push({ ...chunk, similarity: finalScore });
                cursor.continue();
            } else {
                const sortedChunks = chunks.sort((a, b) => b.similarity - a.similarity);
                const selectedChunks = [];
                
                for (const chunk of sortedChunks) {
                    const chunkTokens = estimateTokens(chunk.content);
                    if (totalTokens + chunkTokens <= targetTokens) {
                        selectedChunks.push(chunk);
                        totalTokens += chunkTokens;
                    }
                    if (totalTokens >= targetTokens) break;
                }
                
                const enrichedChunks = selectedChunks.map(chunk => ({
                    ...chunk,
                    text: chunk.content,
                    content: chunk.metadata.precedingContext 
                        ? `${chunk.metadata.precedingContext}\n\n${chunk.content}\n\n${chunk.metadata.followingContext}`.trim()
                        : chunk.content,
                    tokens: estimateTokens(chunk.content)
                }));
                
                resolve(enrichedChunks);
            }
        };
    });
}

async function exportStore() {
    const db = await openEmbeddingsDB();
    const transaction = db.transaction(EMBEDDINGS_STORE_NAME, 'readonly');
    const store = transaction.objectStore(EMBEDDINGS_STORE_NAME);
    
    return new Promise((resolve, reject) => {
        const request = store.getAll();
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
            const data = request.result;
            resolve(new Blob([JSON.stringify(data)], {type: 'application/json'}));
        };
    });
}
async function checkAndLoadLocalKnowledgeBase() {
    try {
        const response = await fetch('https://backup.vdo.ninja/knowledge_base.json');
        if (response.ok) {
            const data = await response.json();
            await importStore(data);
            return true;
        }
    } catch (error) {
        console.log('No local knowledge base found');
    }
    return false;
}
async function importStore(data) {
    const db = await openEmbeddingsDB();
    const transaction = db.transaction(EMBEDDINGS_STORE_NAME, 'readwrite');
    const store = transaction.objectStore(EMBEDDINGS_STORE_NAME);
    
    for (const item of data) {
        await new Promise((resolve, reject) => {
            const request = store.put(item);
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve();
        });
    }
}
async function recursiveQueryRAG(originalQuery, maxQuestions = 5) {
    const searchHistory = [];
    let allRelevantChunks = [];
    
    for (let i = 0; i < maxQuestions; i++) {
        const questionPrompt = buildLLMPrompt(originalQuery, searchHistory);
        const newQuestion = await generateResponse(questionPrompt, []); 
        
        const matches = await searchWithContext(newQuestion, 5, 2);
        if (matches && matches.length > 0) {
            allRelevantChunks = [...allRelevantChunks, ...matches];
            searchHistory.push({
                question: newQuestion,
                matches: matches.map(match => ({
                    content: match.content,
                    context: match.context,
                    similarity: match.similarity
                }))
            });
        }
        
        if (allRelevantChunks.length > 20) break;
    }
    
    const contextText = allRelevantChunks.map(result => `
Previous context:
${result.context?.before || ''}

Relevant text:
${result.content}

Following context:
${result.context?.after || ''}
    `).join('\n---\n');
    
    const finalAnswer = await generateResponse(originalQuery, contextText);
    
    return {
        finalAnswer,
        searchHistory,
        sources: allRelevantChunks.map(chunk => ({
            content: chunk.content,
            context: chunk.context,
            similarity: chunk.similarity,
            metadata: chunk.metadata
        }))
    };
}

function sanitizeMarkdownText(text) {
    return text
        .replace(/\\\\/g, '\\') // Fix escaped backslashes
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove markdown links but keep text
        .replace(/[*_]{1,2}([^*_]+)[*_]{1,2}/g, '$1') // Remove bold/italic markers
		.replace(/[ \t]+/g, ' ')
        .replace(/\n+/g, '\n')
        .replace(/`([^`]+)`/g, '$1')
		.trim(); // Remove code markers
}

function estimateTokens(text) {
    if (!text) return 0;
    
    const patterns = {
        consecutiveWhitespace: /[ \t\n\r]+/g,
        tableSeparators: /\|\s*/g,
        numbers: /\d+/g,
        punctuation: /[.,!?;:'"()\[\]{}]/g,
        specialChars: /[^a-zA-Z0-9\s.,!?;:'"()\[\]{}]/g
    };

    // Normalize consecutive whitespace while preserving newlines
    const normalizedText = text
        .replace(/[ \t]+/g, ' ')
        .replace(/\n+/g, '\n')
        .trim();

    const counts = {
        words: normalizedText.split(/\s+/).length,
        consecutiveWhitespace: (text.match(patterns.consecutiveWhitespace) || [])
            .filter(ws => ws.length > 1).length,
        tableSeparators: (text.match(patterns.tableSeparators) || []).length,
        numbers: (text.match(patterns.numbers) || []).length,
        punctuation: (text.match(patterns.punctuation) || []).length,
        specialChars: (text.match(patterns.specialChars) || []).length,
        newlines: (text.match(/\n/g) || []).length
    };

    const weights = {
        words: 1.3,
        consecutiveWhitespace: 0.1,
        tableSeparators: 0.5,
        numbers: 0.5,
        punctuation: 0.3,
        specialChars: 1.5,
        newlines: 0.2
    };

    let estimate = Object.entries(counts)
        .reduce((total, [key, count]) => total + count * weights[key], 0);

    // Additional weight for UTF-8 characters
    const utf8Length = new TextEncoder().encode(text).length;
    const utf8Weight = utf8Length > text.length ? 1.2 : 1;
    
    estimate *= utf8Weight;

    return Math.ceil(estimate);
}

// Utility function to chunk text
function chunkText(text, maxTokens = 300, metadata = {}) {
    // Initial text preparation
    const sanitizedText = metadata.type === "md" 
        ? sanitizeMarkdownText(text) 
        : text;

    // First split on major break points (headers, bullets, paragraphs)
    let sections = sanitizedText.split(/(?:\n\s*[*•]\s+|\n#{1,6}\s|\n\n+)/);
    const chunks = [];
    
    // Process each section
    for (let section of sections) {
        section = section.trim();
        if (!section) continue;

        // If section already exceeds max, split by sentences
        if (estimateTokens(section) > maxTokens) {
            const sentences = section.split(/(?<=[.!?])\s+/);
            
            let currentChunk = '';
            for (let sentence of sentences) {
                sentence = sentence.trim();
                if (!sentence) continue;

                // If single sentence exceeds max, split by phrases
                if (estimateTokens(sentence) > maxTokens) {
                    const phrases = sentence.split(/(?:[,;:]|\sand\s|\sor\s|\sbut\s)/);
                    
                    for (let phrase of phrases) {
                        phrase = phrase.trim();
                        if (!phrase) continue;

                        // Last resort: split by words if still too large
                        if (estimateTokens(phrase) > maxTokens) {
                            const words = phrase.split(/\s+/);
                            let wordChunk = '';
                            
                            for (let word of words) {
                                if (estimateTokens(wordChunk + ' ' + word) > maxTokens) {
                                    if (wordChunk) chunks.push(wordChunk);
                                    wordChunk = word;
                                } else {
                                    wordChunk = wordChunk 
                                        ? wordChunk + ' ' + word 
                                        : word;
                                }
                            }
                            if (wordChunk) chunks.push(wordChunk);
                            
                        } else {
                            chunks.push(phrase);
                        }
                    }
                } else if (estimateTokens(currentChunk + ' ' + sentence) > maxTokens) {
                    if (currentChunk) chunks.push(currentChunk);
                    currentChunk = sentence;
                } else {
                    currentChunk = currentChunk 
                        ? currentChunk + ' ' + sentence 
                        : sentence;
                }
            }
            if (currentChunk) chunks.push(currentChunk);
            
        } else {
            chunks.push(section);
        }
    }

    // Final verification pass
    return chunks.reduce((verified, chunk) => {
        if (estimateTokens(chunk) <= maxTokens) {
            verified.push(chunk);
        } else {
            // Emergency word-level split if somehow still too large
            const words = chunk.split(/\s+/);
            let emergency = '';
            for (let word of words) {
                if (estimateTokens(emergency + ' ' + word) > maxTokens) {
                    if (emergency) verified.push(emergency);
                    emergency = word;
                } else {
                    emergency = emergency ? emergency + ' ' + word : word;
                }
            }
            if (emergency) verified.push(emergency);
        }
        return verified;
    }, []);
}

function enrichChunkMetadata(chunk, docId, content, index) {
    // Extract key entities and concepts
    const sentences = chunk.split(/[.!?]+\s+/);
    const precedingContext = content.substring(Math.max(0, content.indexOf(chunk) - 200), content.indexOf(chunk));
    const followingContext = content.substring(content.indexOf(chunk) + chunk.length, Math.min(content.indexOf(chunk) + chunk.length + 200));
    
    return {
        docId,
        chunkIndex: index,
        sentenceCount: sentences.length,
        precedingContext: precedingContext.trim(),
        followingContext: followingContext.trim(),
        timestamp: Date.now()
    };
}

// Get embeddings from Ollama
async function getEmbedding(text) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    try {
        const response = await fetch(`${OLLAMA_ENDPOINT}/api/embeddings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: EMBEDDING_MODEL,
                prompt: text
            }),
            signal: controller.signal
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        return data.embedding;
    } catch (error) {
		console.log("estimateTokens: "+estimateTokens(text));
        if (error.name === 'AbortError') {
            throw new Error('Request timeout - check if Ollama is running');
        }
        throw error;
    } finally {
        clearTimeout(timeoutId);
    }
}
// Calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (magnitudeA * magnitudeB);
}

async function getStoreStats() {
    const db = await openEmbeddingsDB();
    const transaction = db.transaction(EMBEDDINGS_STORE_NAME, 'readonly');
    const store = transaction.objectStore(EMBEDDINGS_STORE_NAME);
    
    return new Promise((resolve, reject) => {
        const request = store.count();
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
            const count = request.result;
            store.openCursor().onsuccess = (event) => {
                let totalSize = 0;
                const cursor = event.target.result;
                if (cursor) {
                    totalSize += JSON.stringify(cursor.value).length;
                    cursor.continue();
                } else {
                    resolve({ count, size: (totalSize / 1024 / 1024).toFixed(2) });
                }
            };
        };
    });
}

async function clearStore() {
    const db = await openEmbeddingsDB();
    const transaction = db.transaction(EMBEDDINGS_STORE_NAME, 'readwrite');
    const store = transaction.objectStore(EMBEDDINGS_STORE_NAME);
    
    return new Promise((resolve, reject) => {
        const request = store.clear();
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve();
    });
}

// Process and store document
async function processDocument(docId, content, metadata = {}, progressCallback = null) {
    const isOllamaRunning = await checkOllamaServer();
    if (!isOllamaRunning) {
        throw new Error('Cannot connect to Ollama server - make sure it is running on port 11434');
    }
    
    console.log(`Processing document ${docId} (${content.length} characters)`);
    
    try {
        const chunks = chunkText(content, 250, metadata);
        console.log(`Split into ${chunks.length} chunks`);
        
        if (chunks.length > 0) {
            console.log(`First chunk size: ${chunks[0].length}`);
            console.log(`Last chunk size: ${chunks[chunks.length-1].length}`);
        }
        
        const db = await openEmbeddingsDB();
        
        for (let i = 0; i < chunks.length; i++) {
            const embedding = await getEmbedding(chunks[i]);
            const chunkData = {
                id: `${docId}_${i}`,
                content: chunks[i],
                embedding,
                metadata: {
                    ...metadata,
                    chunkIndex: i,
                    docId,
                    timestamp: Date.now()
                }
            };
            
            // Create new transaction for each chunk
            const transaction = db.transaction(EMBEDDINGS_STORE_NAME, 'readwrite');
            const store = transaction.objectStore(EMBEDDINGS_STORE_NAME);
            
            await new Promise((resolve, reject) => {
                const request = store.put(chunkData);
                transaction.oncomplete = () => resolve();
                transaction.onerror = () => reject(transaction.error);
                request.onerror = () => reject(request.error);
            });
            
            if (progressCallback) {
                progressCallback((i + 1) / chunks.length * 100, docId);
            }
        }
    } catch (error) {
        console.error('Error processing document:', error);
        throw error;
    }
}
// Add near the top of rag.js, after the constants
async function checkOllamaServer() {
    try {
        const response = await fetch(`${OLLAMA_ENDPOINT}/api/tags`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }
        
        const data = await response.json();
        // Check if our required model is available
        const modelExists = data.models?.some(model => 
            model.name.toLowerCase() === EMBEDDING_MODEL.toLowerCase()
        );
        
        if (!modelExists) {
            throw new Error(`Required model '${EMBEDDING_MODEL}' not found`);
        }
        
        return true;
    } catch (error) {
        console.error('Ollama server check failed:', error);
        throw new Error(`Cannot connect to Ollama server: ${error.message}`);
    }
}

async function searchWithContext(query, topK = 5, contextWindow = 2) {
    const queryEmbedding = await getEmbedding(query);
    const db = await openEmbeddingsDB();
    const transaction = db.transaction(EMBEDDINGS_STORE_NAME, 'readonly');
    const store = transaction.objectStore(EMBEDDINGS_STORE_NAME);
    
    return new Promise((resolve, reject) => {
        const allChunks = [];
        
        store.openCursor().onsuccess = (event) => {
            const cursor = event.target.result;
            if (cursor) {
                allChunks.push(cursor.value);
                cursor.continue();
            } else {
                // Group chunks by document
                const docChunks = {};
                allChunks.forEach(chunk => {
                    if (!docChunks[chunk.metadata.docId]) {
                        docChunks[chunk.metadata.docId] = [];
                    }
                    docChunks[chunk.metadata.docId].push(chunk);
                });

                // Score chunks
                const scoredChunks = allChunks.map(chunk => {
                    const similarity = cosineSimilarity(queryEmbedding, chunk.embedding);
                    return { ...chunk, similarity };
                });

                // Get top matches with context
                const topMatches = scoredChunks
                    .sort((a, b) => b.similarity - a.similarity)
                    .slice(0, topK)
                    .map(match => {
                        const doc = docChunks[match.metadata.docId];
                        const chunkIndex = doc.findIndex(c => c.id === match.id);
                        
                        // Get surrounding chunks
                        const start = Math.max(0, chunkIndex - contextWindow);
                        const end = Math.min(doc.length, chunkIndex + contextWindow + 1);
                        const contextChunks = doc.slice(start, end);
                        
                        return {
                            ...match,
                            context: {
                                before: contextChunks.slice(0, contextChunks.indexOf(match)).map(c => c.content).join('\n'),
                                after: contextChunks.slice(contextChunks.indexOf(match) + 1).map(c => c.content).join('\n')
                            }
                        };
                    });

                resolve(topMatches);
            }
        };
    });
}

// Search function
async function searchRAG(query, topK = 5) {
    const queryEmbedding = await getEmbedding(query);
    const db = await openEmbeddingsDB();
    const transaction = db.transaction(EMBEDDINGS_STORE_NAME, 'readonly');
    const store = transaction.objectStore(EMBEDDINGS_STORE_NAME);
    
    return new Promise((resolve, reject) => {
        const chunks = [];
        
        store.openCursor().onsuccess = (event) => {
            const cursor = event.target.result;
            if (cursor) {
                const chunk = cursor.value;
                const semanticScore = cosineSimilarity(queryEmbedding, chunk.embedding);
                const contextScore = chunk.metadata.precedingContext && chunk.metadata.followingContext ? 0.2 : 0;
                const finalScore = semanticScore + contextScore;
                
                chunks.push({ ...chunk, similarity: finalScore });
                cursor.continue();
            } else {
                const topChunks = chunks
                    .sort((a, b) => b.similarity - a.similarity)
                    .slice(0, topK);
                
                const enrichedChunks = topChunks.map(chunk => ({
                    ...chunk,
                    text: chunk.content,
                    content: chunk.metadata.precedingContext 
                        ? `${chunk.metadata.precedingContext}\n\n${chunk.content}\n\n${chunk.metadata.followingContext}`.trim()
                        : chunk.content
                }));
                
                resolve(enrichedChunks);
            }
        };
    });
}

function buildLLMPrompt(originalQuery, searchHistory = [], isFinal = false) {
    if (isFinal) {
        return `Based on all the information gathered, please answer this question: ${originalQuery}

Provide a focused, direct response that synthesizes the relevant information, including contextual details where relevant. Speak as if you are the author of the information.`;
    } 
    
    if (searchHistory.length) {
        const historyContext = searchHistory.map((item, idx) => `
Question ${idx + 1}: ${item.question}
Top Matches:
${item.matches.map((match, i) => `
${i + 1}. Context before: ${match.context?.before || '[none]'}
   Main content: ${match.content}
   Context after: ${match.context?.after || '[none]'}`).join('\n')}`).join('\n\n');

        return `Original question: ${originalQuery}

Previous searches and their results, including context:
${historyContext}

Generate a new search query that would help gather additional relevant information. The query should:
1. Explore aspects not covered by previous searches
2. Help fill knowledge gaps based on existing results
3. Be specific and focused
4. Relate directly to the original question

Return only the search query without any other text.`;
    }
    
    return originalQuery;
}

// Generate response using Ollama
async function generateResponse(query, relevantChunks=[]) {
    const chunks = Array.isArray(relevantChunks) ? relevantChunks : [relevantChunks];
    const context = chunks
        .filter(chunk => chunk && (chunk.text || chunk.content))
        .map(chunk => chunk.text || chunk.content)
        .join('\n\n');

    const prompt = `
Context:
${context || initialContext}

Question: ${query}

Instructions:
1. Answer based on the provided context
2. Be direct and specific
3. Synthesize information from multiple sources if relevant
4. Stay focused on the question

Answer:`;

    try {
        const response = await fetch(`${OLLAMA_ENDPOINT}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: COMPLETION_MODEL,
                prompt: prompt,
                stream: false
            })
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        return data.response;
    } catch (error) {
        console.error('Error generating response:', error);
        throw error;
    }
}

// Main RAG query function
async function queryRAG(query) {
    return recursiveQueryRAG(query);
}

// Handle exports for ES modules
const RAG = {
    processDocument,
    queryRAG,
    openEmbeddingsDB,
    clearStore,
    getStoreStats,
    searchRAGWithTokens,
    recursiveQueryRAG,
    checkAndLoadLocalKnowledgeBase,
    exportStore
};

export default RAG;
