import React, { useState, useEffect, useRef } from 'react';
import WorkflowBar from './WorkflowBar';
import StageDetailPanel from './StageDetailPanel';

// Logo Component
const Logo = () => <span className="font-black text-xl tracking-tight">Lit<span className="text-blue-600">Scout</span></span>;

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

// Sub-component for sidebar
const Sidebar = ({ isOpen, history, onNewChat, onLoadHistory }) => (
  <div className={`${isOpen ? 'w-64' : 'w-0'} bg-gray-900 text-gray-100 transition-all duration-300 flex flex-col border-r border-gray-800 overflow-hidden`}>
    <div className="p-4 border-b border-gray-800 flex justify-between items-center whitespace-nowrap">
       <div className="flex items-center gap-2 text-white"><Logo /></div>
       <button onClick={onNewChat} className="text-xs bg-gray-800 hover:bg-gray-700 px-3 py-1.5 rounded transition-colors">+ New</button>
    </div>
    <div className="overflow-y-auto p-2 flex-grow custom-scrollbar">
       {history.map((item) => (
        <button key={item.id} onClick={() => onLoadHistory(item.id)} className="w-full text-left p-3 rounded hover:bg-gray-800 mb-1 group transition-colors">
          <div className="text-sm truncate text-gray-300 group-hover:text-white">{item.query}</div>
          <div className="text-xs text-gray-600 mt-1">{new Date(item.created_at).toLocaleDateString()}</div>
        </button>
      ))}
    </div>
  </div>
);

function Dashboard() {
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [fullReport, setFullReport] = useState(null); // We keep the state, but won't display it here
  const [workflow, setWorkflow] = useState(INITIAL_WORKFLOW);
  const [selectedStage, setSelectedStage] = useState(null);
  const [history, setHistory] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [terminalLogs, setTerminalLogs] = useState([]); 
  const terminalEndRef = useRef(null);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [terminalLogs]);

  useEffect(() => {
    const saved = localStorage.getItem('litscout_history');
    if (saved) setHistory(JSON.parse(saved));
  }, []);

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

      if (!response.ok) throw new Error("Failed to connect to Python backend");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const jsonString = line.replace("data: ", "").trim();
              if (!jsonString) continue;
              const data = JSON.parse(jsonString);

              if (data.type === "log") {
                setTerminalLogs(prev => [...prev, data.message]);
              } else if (data.type === "final") {
                setFullReport(data.report);
                setWorkflow(data.steps); 
                const newEntry = {
                    id: Date.now(),
                    query: inputValue,
                    report: data.report,
                    workflow: data.steps,
                    created_at: new Date().toISOString()
                };
                const updatedHistory = [newEntry, ...history];
                setHistory(updatedHistory);
                localStorage.setItem('litscout_history', JSON.stringify(updatedHistory));
              }
            } catch (e) {
              console.error("Error parsing stream JSON", e);
            }
          }
        }
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

      <div className="flex-grow flex flex-col h-screen relative bg-white transition-all">
        <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="absolute top-4 left-4 z-20 text-gray-500 hover:text-gray-800 transition-transform p-1 rounded-md hover:bg-gray-100">
           {isSidebarOpen ? '←' : '→'}
        </button>

        <header className="p-4 border-b text-center font-bold text-gray-800 bg-white shadow-sm z-10">Research Orchestrator</header>

        <main className="flex-grow p-6 overflow-y-auto w-full max-w-6xl mx-auto custom-scrollbar">
          
          {/* Welcome Screen */}
          {!loading && !fullReport && terminalLogs.length === 0 && (
            <div className="text-center text-gray-500 mt-20 mb-10">
              <h2 className="text-3xl font-bold mb-2 text-gray-700">Welcome to LitScout</h2>
              <p className="text-lg">Enter a research topic below to begin the autonomous review.</p>
            </div>
          )}

          {/* Terminal View */}
          {(loading || terminalLogs.length > 0) && (
            <div className="mb-8 bg-slate-900 rounded-lg p-4 shadow-2xl border border-slate-700 font-mono text-sm animate-fadeIn">
              <div className="flex items-center gap-2 mb-3 border-b border-slate-800 pb-2">
                <div className="flex gap-1.5"><div className="w-2.5 h-2.5 rounded-full bg-red-500"></div><div className="w-2.5 h-2.5 rounded-full bg-amber-500"></div><div className="w-2.5 h-2.5 rounded-full bg-emerald-500"></div></div>
                <span className="text-slate-500 text-[10px] ml-2 uppercase tracking-widest">Orchestrator Terminal</span>
              </div>
              <div className="h-40 overflow-y-auto custom-scrollbar space-y-1 font-mono text-xs md:text-sm">
                {terminalLogs.map((log, i) => (
                  <div key={i} className="text-emerald-400 leading-relaxed break-words">
                    <span className="text-slate-600 mr-2 select-none">[{new Date().toLocaleTimeString([], {hour12:false})}]</span>
                    <span className="text-blue-500 mr-2 select-none">➜</span>
                    {log}
                  </div>
                ))}
                {loading && <div className="text-white animate-pulse mt-1">▋ Agent active...</div>}
                <div ref={terminalEndRef} />
              </div>
            </div>
          )}

          {/* Workflow Buttons */}
          <WorkflowBar workflow={workflow} selectedStage={selectedStage} onSelectStage={setSelectedStage} />
          
          {/* Detail Panel */}
          <StageDetailPanel stage={selectedStage} onClose={() => setSelectedStage(null)} />
          
          {/* REMOVED: <ReportView /> is deleted. 
             The report text will ONLY show when you click the "Publisher" button 
             because we added PublisherView to StageDetailPanel. */}
        
        </main>

        <footer className="p-4 bg-white border-t z-20">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex gap-4">
            <input 
                className="flex-grow border border-gray-300 rounded-full px-5 py-3 focus:ring-2 focus:ring-blue-500 outline-none shadow-sm transition-all bg-gray-50 focus:bg-white" 
                value={inputValue} onChange={e => setInputValue(e.target.value)} 
                placeholder="Enter research topic..." disabled={loading}
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