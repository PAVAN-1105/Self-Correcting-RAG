"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Loader2, CheckCircle2, ChevronRight, FileText, History } from "lucide-react";

type Step = {
  id: string;
  label: string;
  status: "pending" | "running" | "done";
};

type HistoryItem = {
  id: string;
  query: string;
  answer: string;
  citations: any[];
};

const NODE_LABELS: Record<string, string> = {
  "retrieve": "Retrieving context from Vector DB",
  "grade_documents": "Grading documents for relevance",
  "transform_query": "Rewriting query for better search",
  "web_search": "Searching the web for missing facts",
  "refine_knowledge": "Refining documents into knowledge strips",
  "generate": "Generating response draft",
  "critique_faithfulness": "Fact-checking draft for hallucinations",
  "critique_relevance": "Evaluating answer usefulness",
};

export default function ScragUI() {
  const [query, setQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [steps, setSteps] = useState<Step[]>([]);
  const [finalAnswer, setFinalAnswer] = useState("");
  const [citations, setCitations] = useState<any[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [currentQuery, setCurrentQuery] = useState("");
  
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [steps, finalAnswer]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    const submitQuery = query.trim();
    if (!submitQuery) return;

    if (finalAnswer) {
      setHistory(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          query: currentQuery,
          answer: finalAnswer,
          citations: citations,
        }
      ]);
    }

    setCurrentQuery(submitQuery);
    setIsSearching(true);
    setSteps([{ id: 'init', label: 'Initializing Pipeline & connecting to local models...', status: 'running' }]);
    setFinalAnswer("");
    setCitations([]);
    setQuery("");

    try {
      const response = await fetch("http://localhost:8000/api/v1/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: submitQuery, max_retries: 2 }),
      });

      if (!response.body) throw new Error("No response body");
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      let currentSteps: Step[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n\n');
        
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          
          const dataStr = line.replace('data: ', '');
          if (dataStr === '[DONE]') {
            setIsSearching(false);
            
            // Mark all steps done
            setSteps(prev => prev.map(s => ({ ...s, status: "done" })));
            continue;
          }
          
          try {
            const data = JSON.parse(dataStr);
            
            if (data.node === "FINAL_RESULT") {
              setFinalAnswer(data.answer);
              setCitations(data.citations || []);
            } else if (data.node) {
              const label = NODE_LABELS[data.node] || `Running step: ${data.node}`;
              const stepId = `${data.node}-${data.retry_count}`;
              
              currentSteps = currentSteps.map(s => s.status === 'running' ? { ...s, status: 'done' } : s);
              
              if (!currentSteps.find(s => s.id === stepId)) {
                currentSteps.push({ id: stepId, label, status: 'running' });
              }
              
              setSteps([...currentSteps]);
            }
          } catch (e) {
            console.error("Error parsing stream chunk", e);
          }
        }
      }
    } catch (error) {
      console.error("Error during search:", error);
      setFinalAnswer("An error occurred while connecting to the pipeline.");
      setIsSearching(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 font-sans selection:bg-zinc-200 dark:selection:bg-zinc-800 flex flex-col">
      
      {/* Header */}
      <header className="w-full max-w-3xl mx-auto py-8 px-6 flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800/50">
        <h1 className="text-xl font-medium tracking-tight flex items-center gap-2">
          SCRAG
        </h1>
        <button
          onClick={() => setShowHistory(!showHistory)}
          className="flex items-center gap-2 text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
        >
          <History className="w-4 h-4" />
          History
        </button>
      </header>

      {/* History Dropdown */}
      <AnimatePresence>
        {showHistory && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="w-full max-w-3xl mx-auto overflow-hidden bg-zinc-100/50 dark:bg-zinc-900/50 border-b border-zinc-200 dark:border-zinc-800/50"
          >
            <div className="p-6 flex flex-col gap-8 max-h-[60vh] overflow-y-auto">
              {history.length === 0 ? (
                <div className="text-center text-sm text-zinc-500 py-8">No history yet</div>
              ) : (
                history.slice(-3).reverse().map((item) => (
                  <div key={item.id} className="flex flex-col gap-4 pb-8 border-b border-zinc-200 dark:border-zinc-800/50 last:border-0 last:pb-0">
                    <div className="text-lg font-medium text-zinc-900 dark:text-zinc-100">
                      {item.query}
                    </div>
                    <div className="prose prose-sm prose-zinc dark:prose-invert max-w-none">
                      <div className="text-sm leading-relaxed whitespace-pre-wrap">
                        {item.answer}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="flex-1 w-full max-w-3xl mx-auto px-6 py-12 flex flex-col">
        
        {/* Search Input */}
        <form onSubmit={handleSearch} className="relative mb-12">
          <div className="relative flex items-center w-full bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl shadow-sm transition-all focus-within:border-zinc-400 dark:focus-within:border-zinc-600 focus-within:ring-4 focus-within:ring-zinc-100 dark:focus-within:ring-zinc-800/50 overflow-hidden">
            <Search className="w-5 h-5 text-zinc-400 ml-4 shrink-0" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question..."
              className="w-full bg-transparent border-none text-base text-zinc-900 dark:text-zinc-100 placeholder-zinc-500 focus:outline-none px-4 py-4"
              disabled={isSearching}
            />
          </div>
        </form>

        {/* Output Area */}
        <div className="flex flex-col gap-8">
          
          {/* History removed from inline view */}

          {/* Current Query */}
          {currentQuery && (
            <div className="text-xl font-medium text-zinc-900 dark:text-zinc-100 mb-2">
              {currentQuery}
            </div>
          )}

          {/* Steps Trace */}
          <AnimatePresence>
            {steps.length > 0 && (
              <motion.div 
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="flex flex-col gap-3 p-6 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl text-sm"
              >
                <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-2">Execution Trace</h3>
                {steps.map((step) => (
                  <motion.div 
                    key={step.id} 
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-center gap-3 text-zinc-600 dark:text-zinc-400"
                  >
                    {step.status === 'running' ? (
                      <Loader2 className="w-4 h-4 animate-spin text-zinc-900 dark:text-zinc-100" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 text-zinc-400 dark:text-zinc-600" />
                    )}
                    <span className={step.status === 'running' ? 'text-zinc-900 dark:text-zinc-100 font-medium' : ''}>
                      {step.label}
                    </span>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Final Answer */}
          <AnimatePresence>
            {finalAnswer && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="prose prose-zinc dark:prose-invert max-w-none"
              >
                <div className="text-base leading-relaxed whitespace-pre-wrap">
                  {finalAnswer}
                </div>
                
                {/* Citations block */}
                {citations.length > 0 && (
                  <div className="mt-10 pt-8 border-t border-zinc-200 dark:border-zinc-800">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-4 flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      Sources
                    </h3>
                    <div className="flex flex-col gap-3">
                      {citations.map((cite, idx) => (
                        <div key={idx} className="text-sm p-4 bg-zinc-100 dark:bg-zinc-900/50 rounded-lg text-zinc-600 dark:text-zinc-400">
                          <span className="font-mono text-xs px-1.5 py-0.5 bg-zinc-200 dark:bg-zinc-800 rounded mr-2">[{idx + 1}]</span>
                          {cite.content}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
          
        </div>
      </main>
    </div>
  );
}
