import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

// Test the complete production module. All model and IndexedDB I/O stays offline.
const source = await readFile(new URL(process.env.RAG_SOURCE || '../rag.js', import.meta.url), 'utf8');
const { default: RAG } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

function chunk(content, id = 'doc_0', embedding = [1, 0]) {
    return { id, content, embedding, metadata: { docId: id.split('_')[0], chunkIndex: 0 } };
}

async function withIO(rows, callback, options = {}) {
    const originalFetch = globalThis.fetch;
    const originalIndexedDB = globalThis.indexedDB;
    const requests = [];
    let round = 0;
    globalThis.indexedDB = {
        open() {
            const request = {};
            request.result = {
                transaction(name, mode) {
                    assert.equal(name, 'embeddings');
                    assert.equal(mode, 'readonly');
                    return {
                        objectStore() {
                            return {
                                openCursor() {
                                    const result = {};
                                    const data = typeof rows === 'function' ? rows(round++) : rows;
                                    let position = 0;
                                    const advance = () => queueMicrotask(() => result.onsuccess({
                                        target: { result: position < data.length ? {
                                            value: structuredClone(data[position++]), continue: advance
                                        } : null }
                                    }));
                                    advance();
                                    return result;
                                }
                            };
                        }
                    };
                }
            };
            queueMicrotask(() => request.onsuccess());
            return request;
        }
    };
    globalThis.fetch = async (url, init) => {
        const body = JSON.parse(init.body);
        requests.push({ url, body });
        if (url === 'http://localhost:11434/api/embeddings') {
            if (options.embeddingFailure) return { ok: false, status: 503 };
            return { ok: true, json: async () => ({ embedding: [1, 0] }) };
        }
        assert.equal(url, 'http://localhost:11434/api/generate');
        if (options.generationFailure) return { ok: false, status: 502 };
        return { ok: true, json: async () => ({ response: options.response || 'model-response' }) };
    };
    try {
        return await callback(requests);
    } finally {
        globalThis.fetch = originalFetch;
        if (originalIndexedDB === undefined) delete globalThis.indexedDB;
        else globalThis.indexedDB = originalIndexedDB;
    }
}

const finalPrompt = requests => requests.filter(r => r.url.endsWith('/api/generate')).at(-1).body.prompt;

test('public queryRAG includes the retrieved source in the final answer prompt', async () => {
    await withIO([chunk('SOURCE_FACT: the test studio switch is violet.')], async requests => {
        const result = await RAG.queryRAG('What color is the test studio switch?');
        assert.match(finalPrompt(requests), /SOURCE_FACT: the test studio switch is violet\./);
        assert.match(finalPrompt(requests), /Question: What color is the test studio switch\?/);
        assert.equal(result.finalAnswer, 'model-response');
        assert.equal(result.searchHistory.length, 5);
    });
});

test('retrieved content replaces the unrelated generic fallback', async () => {
    await withIO([chunk('UNIQUE_PROJECT_FACT_123')], async requests => {
        await RAG.recursiveQueryRAG('A custom knowledge base question', 1);
        const context = finalPrompt(requests).split('Context:\n')[1].split('\n\nQuestion:')[0];
        assert.match(context, /UNIQUE_PROJECT_FACT_123/);
        assert.doesNotMatch(context, /VDO\.Ninja is a free, open-source web service/);
    });
});

test('final prompt includes results accumulated over separate retrieval rounds', async () => {
    await withIO(round => [chunk(`ROUND_${round}_FACT`, `doc${round}_0`)], async requests => {
        const result = await RAG.recursiveQueryRAG('A multi-part question', 3);
        for (let i = 0; i < 3; i++) assert.ok(finalPrompt(requests).includes(`ROUND_${i}_FACT`));
        assert.equal(result.sources.length, 3);
    });
});

test('final prompt keeps selected results from multiple documents', async () => {
    await withIO([chunk('DOCUMENT_A_FACT', 'a_0'), chunk('DOCUMENT_B_FACT', 'b_0')], async requests => {
        await RAG.recursiveQueryRAG('Compare the documents', 1);
        assert.match(finalPrompt(requests), /DOCUMENT_A_FACT/);
        assert.match(finalPrompt(requests), /DOCUMENT_B_FACT/);
    });
});

test('Unicode and multiline retrieved content survives unchanged', async () => {
    const text = 'FACT: café 東京\nsecond line: 🟣';
    await withIO([chunk(text)], async requests => {
        await RAG.recursiveQueryRAG('Read the source', 1);
        assert.ok(finalPrompt(requests).includes(text));
    });
});

test('empty retrieval keeps the original generic-context fallback', async () => {
    await withIO([], async requests => {
        const result = await RAG.recursiveQueryRAG('No matches', 1);
        assert.match(finalPrompt(requests), /VDO\.Ninja is a free, open-source web service/);
        assert.deepEqual(result.sources, []);
        assert.deepEqual(result.searchHistory, []);
    });
});

test('zero recursive rounds keeps fallback and makes no embedding request', async () => {
    await withIO([], async requests => {
        await RAG.recursiveQueryRAG('No retrieval requested', 0);
        assert.equal(requests.length, 1);
        assert.match(finalPrompt(requests), /VDO\.Ninja is a free, open-source web service/);
    });
});

test('subsequent search-query prompt still contains earlier search history', async () => {
    await withIO([chunk('HISTORY_ONLY_MARKER')], async requests => {
        await RAG.recursiveQueryRAG('Follow up', 2);
        const generations = requests.filter(r => r.url.endsWith('/api/generate'));
        assert.match(generations[1].body.prompt, /HISTORY_ONLY_MARKER/);
        assert.match(generations[1].body.prompt, /Previous searches and their results/);
    });
});

test('source attribution and model response are preserved', async () => {
    const record = chunk('Attribution source', 'manual_0');
    await withIO([record], async () => {
        const result = await RAG.recursiveQueryRAG('Attribution', 1);
        assert.equal(result.finalAnswer, 'final answer unchanged');
        assert.equal(result.sources[0].content, record.content);
        assert.deepEqual(result.sources[0].metadata, record.metadata);
    }, { response: 'final answer unchanged' });
});

test('embedding failure still rejects without attempting final answer', async () => {
    await withIO([], async requests => {
        await assert.rejects(RAG.recursiveQueryRAG('Failed embedding', 1), /HTTP error! status: 503/);
        assert.equal(requests.filter(r => r.url.endsWith('/api/generate')).length, 1);
    }, { embeddingFailure: true });
});

test('generation failure still rejects', async () => {
    await withIO([], async () => {
        await assert.rejects(RAG.recursiveQueryRAG('Failed model', 1), /HTTP error! status: 502/);
    }, { generationFailure: true });
});
