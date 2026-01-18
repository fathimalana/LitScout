import React, { useState, useEffect, useRef } from 'react';
import Logo from './Logo';

// --- 1. INITIAL STATE (Empty Buttons) ---
const INITIAL_WORKFLOW = [
    { id: "stage_1", name: "Research Planner", role: "Strategy", status: "idle", stats: "Waiting...", data: {} },
    { id: "stage_2", name: "Deep Crawler", role: "Search & Filter", status: "idle", stats: "Waiting...", data: {} },
    { id: "stage_3", name: "Semantic Screen", role: "Screening", status: "idle", stats: "Waiting...", data: {} },
    { id: "stage_4", name: "Data Extractor", role: "Extraction", status: "idle", stats: "Waiting...", data: {} },
    { id: "stage_5", name: "Theme Mapper", role: "Thematic Analysis", status: "idle", stats: "Waiting...", data: {} },
    { id: "stage_6", name: "Synthesizer", role: "Drafting", status: "idle", stats: "Waiting...", data: {} },
    { id: "stage_7", name: "QA Bot", role: "Quality Assurance", status: "idle", stats: "Waiting...", data: {} },
    { id: "stage_8", name: "Publisher", role: "Final Report", status: "idle", stats: "Waiting...", data: {} }
];

// --- 2. UI COMPONENTS ---

const Sidebar = ({ isOpen, history, onNewChat, onLoadHistory }) => (
  <div className={`${isOpen ? 'w-64' : 'w-0'} bg-gray-900 text-gray-100 transition-all duration-300 flex flex-col border-r border-gray-800 overflow-hidden`}>
    <div className="p-4 border-b border-gray-800 flex justify-between items-center whitespace-nowrap">
       <div className="flex items-center">
          <Logo className="h-8 w-auto" />
       </div>
       <button onClick={onNewChat} className="text-xs bg-gray-800 p-2 rounded hover:bg-gray-700">+ New</button>
    </div>
    <div className="overflow-y-auto p-2 flex-grow">
       {history.map((item) => (
        <button key={item.id} onClick={() => onLoadHistory(item.id)} className="w-full text-left p-3 rounded hover:bg-gray-800 mb-1 group">
          <div className="text-sm truncate text-gray-300 group-hover:text-white">{item.query}</div>
          <div className="text-xs text-gray-600">{new Date(item.created_at).toLocaleDateString()}</div>
        </button>
      ))}
    </div>
  </div>
);

const WorkflowBar = ({ workflow, selectedStage, onSelectStage }) => (
    <div className="mb-8 animate-fadeIn">
        <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Agent Workflow Pipeline</h3>
        <div className="flex space-x-3 overflow-x-auto pb-4 custom-scrollbar">
            {workflow.map((stage) => (
                <button 
                    key={stage.id}
                    onClick={() => onSelectStage(stage)}
                    className={`flex-shrink-0 flex flex-col justify-between p-3 rounded-lg border transition-all w-40 h-28 text-left
                        ${selectedStage?.id === stage.id 
                            ? 'bg-blue-600 text-white border-blue-600 shadow-lg scale-105' 
                            : stage.status === 'idle' 
                                ? 'bg-gray-50 border-gray-200 text-gray-400' 
                                : 'bg-white border-gray-200 text-gray-600 hover:border-blue-300 hover:shadow-md'
                        }`}
                >
                    <div>
                        <div className={`text-[10px] uppercase font-bold tracking-wider mb-1 ${selectedStage?.id === stage.id ? 'text-blue-200' : 'text-gray-400'}`}>
                            {stage.role}
                        </div>
                        <div className="font-bold text-sm leading-tight">{stage.name}</div>
                    </div>
                    <div className={`text-[10px] mt-2 ${selectedStage?.id === stage.id ? 'text-blue-100' : stage.status === 'completed' ? 'text-green-600' : 'text-gray-400'}`}>
                        {stage.stats}
                    </div>
                </button>
            ))}
        </div>
    </div>
);

const StageDetailPanel = ({ stage, onClose }) => {
  if (!stage) return null;
  return (
    <div className="bg-gray-800 rounded-xl p-6 text-gray-300 shadow-inner mb-6 animate-fadeIn border border-gray-700">
      <div className="flex justify-between items-center border-b border-gray-600 pb-3 mb-4">
          <h3 className="text-white font-mono font-bold flex items-center">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
              Output: {stage.name}
          </h3>
          <button onClick={onClose} className="text-xs text-gray-400 hover:text-white">✕ CLOSE</button>
      </div>
      <div className="font-mono text-sm space-y-4">
          {stage.status === 'idle' && (
              <div className="text-gray-500 italic text-center p-4">Waiting for research to start...</div>
          )}
          {stage.role === "Strategy" && stage.status !== 'idle' && (
              <>
                  <div>
                      <span className="text-blue-400 font-bold block mb-1">Research Questions:</span>
                      <ul className="list-disc pl-5 text-gray-300">
                          {stage.data.questions?.map((q, i) => <li key={i}>{q}</li>)}
                      </ul>
                  </div>
                  <div className="grid grid-cols-2 gap-4 mt-2">
                      <div className="bg-gray-900 p-2 rounded">
                          <span className="text-green-400 font-bold block text-xs">INCLUSION</span>
                          {stage.data.criteria?.inclusion}
                      </div>
                      <div className="bg-gray-900 p-2 rounded">
                          <span className="text-red-400 font-bold block text-xs">EXCLUSION</span>
                          {stage.data.criteria?.exclusion}
                      </div>
                  </div>
              </>
          )}
          {stage.role === "Search & Filter" && stage.status !== 'idle' && (
              <div className="space-y-4">
                  <div className="flex justify-between bg-gray-900 p-3 rounded-lg border border-gray-600">
                       <div className="text-center"><div className="text-gray-400 text-xs uppercase">Total Found</div><div className="text-white text-xl font-bold">{stage.data.summary?.total || 0}</div></div>
                       <div className="text-center"><div className="text-gray-400 text-xs uppercase">Excluded</div><div className="text-red-400 text-xl font-bold">{stage.data.summary?.rejected || 0}</div></div>
                       <div className="text-center border-l border-gray-600 pl-4"><div className="text-green-400 text-xs uppercase font-bold">Selected</div><div className="text-green-400 text-xl font-bold">{stage.data.summary?.selected || 0}</div></div>
                  </div>
              </div>
          )}
          {!['Strategy', 'Search & Filter'].includes(stage.role) && stage.status !== 'idle' && (
              <div className="text-gray-500 italic">Data view for {stage.name} is coming soon.</div>
          )}
      </div>
    </div>
  );
};

const ReportView = ({ reportText, loading }) => {
  const [displayedText, setDisplayedText] = useState('');
  const typewriterRef = useRef(null);

  useEffect(() => {
    if (reportText && !loading) {
      let index = 0;
      setDisplayedText('');
      clearInterval(typewriterRef.current);
      typewriterRef.current = setInterval(() => {
        setDisplayedText((prev) => {
            if (index < reportText.length) {
                index++;
                return reportText.slice(0, index);
            }
            clearInterval(typewriterRef.current);
            return prev;
        });
      }, 5);
    }
    return () => clearInterval(typewriterRef.current);
  }, [reportText, loading]);

  if (!reportText) return null;

  return (
    <div className="bg-white p-10 rounded-xl shadow-xl border border-gray-100 animate-fadeIn mb-10">
       <div className="prose lg:prose-xl max-w-none text-gray-800 font-serif whitespace-pre-wrap leading-relaxed">
         {displayedText}
         <span className="inline-block w-2 h-5 bg-blue-500 animate-pulse ml-1 align-middle"></span>
       </div>
    </div>
  );
};

// --- 3. MAIN DASHBOARD CONTROLLER ---

function Dashboard() {
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [fullReport, setFullReport] = useState(null);
  const [workflow, setWorkflow] = useState(INITIAL_WORKFLOW);
  const [selectedStage, setSelectedStage] = useState(null);
  const [history, setHistory] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // --- NEW: TERMINAL LOGS STATE ---
  const [terminalLogs, setTerminalLogs] = useState([]);
  const terminalEndRef = useRef(null);

  useEffect(() => {
    const saved = localStorage.getItem('litscout_history');
    if (saved) setHistory(JSON.parse(saved));
  }, []);

  // Auto-scroll terminal
  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [terminalLogs]);

  const startNewChat = () => {
    setFullReport(null);
    setInputValue('');
    setWorkflow(INITIAL_WORKFLOW);
    setSelectedStage(null);
    setTerminalLogs([]);
  };

  const loadHistoryItem = (id) => {
    setLoading(true);
    setWorkflow(INITIAL_WORKFLOW);
    setSelectedStage(null);
    setTerminalLogs([]);
    
    setTimeout(() => {
        const item = history.find(h => h.id === id);
        if (item) {
            setFullReport(item.report);
            setWorkflow(item.workflow || []);
            setInputValue(item.query);
        }
        setLoading(false);
    }, 500);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!inputValue.trim()) return;
    
    setLoading(true);
    setFullReport(null);
    setSelectedStage(null);
    setTerminalLogs(["📡 Connecting to LitScout Orchestrator..."]);
    setWorkflow(INITIAL_WORKFLOW);

    try {
      const response = await fetch('http://localhost:8000/start-research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: inputValue }),
      });

      if (!response.ok) throw new Error("Backend connection failed");

      // --- SSE STREAM READER ---
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n\n");

        lines.forEach(line => {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.replace("data: ", ""));
              
              if (data.type === "log") {
                setTerminalLogs(prev => [...prev, data.message]);
              } else if (data.type === "final") {
                setFullReport(data.report);
                setWorkflow(data.steps);
                setSelectedStage(data.steps[0]);
                
                // Save to history
                const newEntry = {
                    id: Date.now(),
                    query: inputValue,
                    report: data.report,
                    workflow: data.steps,
                    created_at: new Date().toISOString()
                };
                setHistory(prev => {
                    const updated = [newEntry, ...prev];
                    localStorage.setItem('litscout_history', JSON.stringify(updated));
                    return updated;
                });
              }
            } catch (e) {
              console.error("Chunk parse error", e);
            }
          }
        });
      }
    } catch (err) {
      setTerminalLogs(prev => [...prev, `🛑 Error: ${err.message}`]);
      setFullReport(`Error: ${err.message}.`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 font-sans overflow-hidden">
      <Sidebar isOpen={isSidebarOpen} history={history} onNewChat={startNewChat} onLoadHistory={loadHistoryItem} />
      <div className="flex-grow flex flex-col h-screen relative bg-white">
        <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="absolute top-4 left-4 z-20 text-gray-500 hover:text-gray-800 transition-transform">
           {isSidebarOpen ? '←' : '→'}
        </button>

        <header className="p-4 border-b text-center font-bold text-gray-800 bg-white">Research Orchestrator</header>

        <main className="flex-grow p-6 overflow-y-auto w-full max-w-6xl mx-auto custom-scrollbar">
          
          {/* Welcome Text */}
          {!loading && !fullReport && terminalLogs.length === 0 && (
            <div className="text-center text-gray-500 mt-10 mb-10 animate-fadeIn">
              <h2 className="text-2xl font-semibold">Welcome to LitScout</h2>
              <p>Enter a research topic below to begin the autonomous review.</p>
            </div>
          )}

          {/* --- TERMINAL LOGS SECTION --- */}
          {(loading || terminalLogs.length > 0) && (
            <div className="mb-8 bg-slate-900 rounded-lg p-4 shadow-2xl border border-slate-700 font-mono text-sm animate-fadeIn">
              <div className="flex items-center gap-2 mb-3 border-b border-slate-800 pb-2">
                <div className="flex gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-amber-500"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500"></div>
                </div>
                <span className="text-slate-500 text-[10px] ml-2 uppercase tracking-widest">Orchestrator Terminal</span>
              </div>
              <div className="h-40 overflow-y-auto custom-scrollbar space-y-1">
                {terminalLogs.map((log, i) => (
                  <div key={i} className="text-emerald-400 leading-relaxed">
                    <span className="text-slate-600 mr-2">[{new Date().toLocaleTimeString([], {hour12:false})}]</span>
                    <span className="text-blue-500 mr-2">➜</span>
                    {log}
                  </div>
                ))}
                {loading && <div className="text-white animate-pulse mt-1">▋ Agent active...</div>}
                <div ref={terminalEndRef} />
              </div>
            </div>
          )}

          <WorkflowBar workflow={workflow} selectedStage={selectedStage} onSelectStage={setSelectedStage} />
          <StageDetailPanel stage={selectedStage} onClose={() => setSelectedStage(null)} />
          {!loading && <ReportView reportText={fullReport} loading={loading} />}
        </main>

        <footer className="p-4 bg-white border-t">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex gap-4">
            <input 
                className="flex-grow border border-gray-300 rounded-full px-5 py-3 focus:ring-2 focus:ring-blue-500 outline-none shadow-sm transition-all bg-gray-50 focus:bg-white" 
                value={inputValue} 
                onChange={e => setInputValue(e.target.value)} 
                placeholder="Enter research topic..." 
                disabled={loading}
            />
            <button type="submit" disabled={loading} className="bg-blue-600 text-white px-8 py-3 rounded-full hover:bg-blue-700 disabled:bg-gray-400 font-medium transition-all shadow-md hover:shadow-lg active:scale-95">
                {loading ? "Orchestrating..." : "Start Research"}
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
}

export default Dashboard;