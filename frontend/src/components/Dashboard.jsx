import React, { useState, useEffect, useRef } from 'react';
import Logo from './Logo';
// --- 1. INITIAL STATE (Empty Buttons) ---
// This ensures the buttons are always visible, even before searching.
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
  <div className={`${isOpen ? 'w-64' : 'w-0'} bg-gray-900 text-gray-100 transition-all duration-300 flex flex-col border-r border-gray-800`}>
    
    {/* HEADER SECTION */}
    <div className="p-4 border-b border-gray-800 flex justify-between items-center">
       
       {/* 2. REPLACED THE TEXT SPAN WITH THIS LOGO COMPONENT */}
       <div className="flex items-center">
          <Logo className="h-8 w-auto" />
       </div>

       <button onClick={onNewChat} className="text-xs bg-gray-800 p-2 rounded hover:bg-gray-700">+ New</button>
    </div>

    <div className="overflow-y-auto p-2 flex-grow">
       {history.map((item) => (
        <button key={item.id} onClick={() => onLoadHistory(item.id)} className="w-full text-left p-3 rounded hover:bg-gray-800 mb-1">
          <div className="text-sm truncate text-gray-300 group-hover:text-white">{item.query}</div>
          <div className="text-xs text-gray-600">{new Date(item.created_at).toLocaleDateString()}</div>
        </button>
      ))}
    </div>
  </div>
);

const WorkflowBar = ({ workflow, selectedStage, onSelectStage }) => {
  return (
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
                                ? 'bg-gray-50 border-gray-200 text-gray-400' // Idle/Empty Look
                                : 'bg-white border-gray-200 text-gray-600 hover:border-blue-300 hover:shadow-md' // Active/Done Look
                        }`}
                >
                    <div>
                        <div className={`text-[10px] uppercase font-bold tracking-wider mb-1 ${selectedStage?.id === stage.id ? 'text-blue-200' : 'text-gray-400'}`}>
                            {stage.role}
                        </div>
                        <div className="font-bold text-sm leading-tight">
                            {stage.name}
                        </div>
                    </div>
                    <div className={`text-[10px] mt-2 ${selectedStage?.id === stage.id ? 'text-blue-100' : stage.status === 'completed' ? 'text-green-600' : 'text-gray-400'}`}>
                        {stage.stats}
                    </div>
                </button>
            ))}
        </div>
    </div>
  );
};

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
          
          {/* HANDLE IDLE STATE */}
          {stage.status === 'idle' && (
              <div className="text-gray-500 italic text-center p-4">
                  Waiting for research to start... <br/>
                  Enter a topic above to activate this agent.
              </div>
          )}

          {/* 1. STRATEGY VIEW */}
          {stage.role === "Strategy" && stage.status !== 'idle' && (
              <>
                  <div>
                      <span className="text-blue-400 font-bold block mb-1">Research Questions:</span>
                      <ul className="list-disc pl-5 text-gray-300">
                          {stage.data.questions && stage.data.questions.map((q, i) => <li key={i}>{q}</li>)}
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

          {/* 2. SEARCH & FILTER VIEW */}
          {stage.role === "Search & Filter" && stage.status !== 'idle' && (
              <div className="space-y-4">
                  <div className="flex justify-between bg-gray-900 p-3 rounded-lg border border-gray-600">
                       <div className="text-center"><div className="text-gray-400 text-xs uppercase">Total Found</div><div className="text-white text-xl font-bold">{stage.data.summary?.total || 0}</div></div>
                       <div className="text-center"><div className="text-gray-400 text-xs uppercase">Excluded</div><div className="text-red-400 text-xl font-bold">{stage.data.summary?.rejected || 0}</div></div>
                       <div className="text-center border-l border-gray-600 pl-4"><div className="text-green-400 text-xs uppercase font-bold">Selected</div><div className="text-green-400 text-xl font-bold">{stage.data.summary?.selected || 0}</div></div>
                  </div>
                  {/* If you have paper lists from backend, render them here */}
                  {stage.data.papers && stage.data.papers.length > 0 ? (
                      <div className="bg-gray-900 rounded border border-gray-700 max-h-60 overflow-y-auto custom-scrollbar">
                        <table className="w-full text-left text-xs">
                            <thead className="bg-gray-800 text-gray-400 sticky top-0"><tr><th className="p-2">Title</th><th className="p-2 text-right">Result</th></tr></thead>
                            <tbody className="divide-y divide-gray-800">
                                {stage.data.papers.map((p, i) => (
                                    <tr key={i} className="hover:bg-gray-800/50">
                                        <td className="p-2 text-gray-300 truncate max-w-xs">{p.title}</td>
                                        <td className="p-2 text-right"><span className={`px-2 py-1 rounded text-[10px] ${p.status === 'Selected' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>{p.status}</span></td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                      </div>
                  ) : (
                      <div className="text-gray-500 italic text-xs">Detailed paper list pending...</div>
                  )}
              </div>
          )}

          {/* FALLBACK FOR OTHER STAGES */}
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
      }, 10);
    }
    return () => clearInterval(typewriterRef.current);
  }, [reportText, loading]);

  if (!reportText) return null;

  return (
    <div className="bg-white p-10 rounded-xl shadow-xl border border-gray-100 animate-fadeIn">
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
  
  // Initialize with EMPTY buttons (Visual Only)
  const [workflow, setWorkflow] = useState(INITIAL_WORKFLOW);
  
  const [selectedStage, setSelectedStage] = useState(null);
  const [history, setHistory] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Load History
  useEffect(() => {
    const saved = localStorage.getItem('litscout_history');
    if (saved) setHistory(JSON.parse(saved));
  }, []);

  const startNewChat = () => {
    setFullReport(null);
    setInputValue('');
    setWorkflow(INITIAL_WORKFLOW); // Reset to empty buttons
    setSelectedStage(null);
  };

  const loadHistoryItem = (id) => {
    setLoading(true);
    setWorkflow(INITIAL_WORKFLOW); // Reset first
    setSelectedStage(null);
    
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
    // Keep the "Empty" buttons visible, maybe mark them as "Processing..." later if desired
    // setWorkflow(INITIAL_WORKFLOW); 
    setSelectedStage(null);

    try {
      const response = await fetch('http://localhost:8000/start-research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: inputValue }),
      });

      if (!response.ok) {
        throw new Error("Failed to connect to Python backend");
      }

      const data = await response.json();

      setFullReport(data.report);
      setWorkflow(data.workflow); // Replace empty buttons with REAL data buttons
      
      const newEntry = {
          id: Date.now(),
          query: inputValue,
          report: data.report,
          workflow: data.workflow,
          created_at: new Date().toISOString()
      };
      
      const updatedHistory = [newEntry, ...history];
      setHistory(updatedHistory);
      localStorage.setItem('litscout_history', JSON.stringify(updatedHistory));

    } catch (err) {
      console.error(err);
      setFullReport(`Error: ${err.message}.`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 font-sans overflow-hidden">
      
      {/* SIDEBAR */}
      <Sidebar 
        isOpen={isSidebarOpen} 
        history={history} 
        onNewChat={startNewChat} 
        onLoadHistory={loadHistoryItem} 
      />

      {/* MAIN CONTENT */}
      <div className="flex-grow flex flex-col h-screen relative bg-white">
        
        <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="absolute top-4 left-4 z-20 text-gray-500 hover:text-gray-800">
           {isSidebarOpen ? '←' : '→'}
        </button>

        <header className="p-4 border-b text-center font-bold text-gray-800 bg-white">
            Research Orchestrator
        </header>

        <main className="flex-grow p-6 overflow-y-auto w-full max-w-6xl mx-auto custom-scrollbar">
          
          {/* Welcome Screen (Only if no report and not loading) */}
          {!loading && !fullReport && (
            <div className="text-center text-gray-500 mt-10 mb-10">
              <h2 className="text-2xl font-semibold">Welcome to LitScout</h2>
              <p>Enter a research topic below.</p>
            </div>
          )}

          {/* Loading Spinner */}
          {loading && (
             <div className="flex flex-col items-center justify-center mb-8 space-y-4">
               <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
               <p className="text-gray-500 animate-pulse">Orchestrating agents...</p>
             </div>
          )}

          {/* --- ALWAYS VISIBLE WORKFLOW BAR --- */}
          <WorkflowBar 
              workflow={workflow} 
              selectedStage={selectedStage} 
              onSelectStage={setSelectedStage} 
          />

          {/* DETAIL PANEL */}
          <StageDetailPanel 
            stage={selectedStage} 
            onClose={() => setSelectedStage(null)} 
          />

          {/* REPORT VIEW */}
          {!loading && (
            <ReportView 
                reportText={fullReport} 
                loading={loading} 
            />
          )}

        </main>

        <footer className="p-4 bg-white border-t">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex gap-4">
            <input 
                className="flex-grow border border-gray-300 rounded-full px-5 py-3 focus:ring-2 focus:ring-blue-500 outline-none shadow-sm transition-all" 
                value={inputValue} 
                onChange={e => setInputValue(e.target.value)} 
                placeholder="Enter research topic..." 
                disabled={loading}
            />
            <button type="submit" disabled={loading} className="bg-blue-600 text-white px-6 py-3 rounded-full hover:bg-blue-700 disabled:bg-gray-400 font-medium transition-all shadow-md hover:shadow-lg">
                Start Research
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
}

export default Dashboard;