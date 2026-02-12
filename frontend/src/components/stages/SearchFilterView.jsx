import React, { useState, useEffect } from 'react';

const SearchFilterView = ({ data }) => {
  const [filterMode, setFilterMode] = useState('ALL'); // 'ALL', 'KEPT', 'REJECTED'

  // Get papers list (safe default)
  const papers = data?.papers || [];
  const summary = data?.summary || {};

  // --- DEBUGGING: Open F12 in Chrome to see this ---
  useEffect(() => {
    console.log("🔍 SearchFilterView Data Received:", data);
    if (papers.length > 0) {
      console.log("📄 First Paper Status:", papers[0].title, "->", papers[0].status);
    }
  }, [data]);

  // --- LOGIC: DEFINE WHAT COUNTS AS "KEPT" ---
  // Inside SearchFilterView.jsx
    const isPaperKept = (status) => {
    if (!status) return true; // Default to green if missing
    const s = status.toLowerCase().trim();
    return ['selected', 'kept', 'passed', 'relevant'].includes(s);
    };

    const isPaperRejected = (status) => {
    if (!status) return false;
    return status.toLowerCase().trim() === 'rejected';
    };
  
  
  return (
    <div className="space-y-4 animate-fadeIn">
        {/* Summary Statistics */}
        <div className="flex justify-between bg-gray-50 p-4 rounded-lg border border-gray-200 shadow-sm">
             <div className="text-center">
                <div className="text-gray-400 text-[10px] uppercase font-bold tracking-wider">Total Scanned</div>
                <div className="text-gray-800 text-2xl font-bold">{summary.total || 0}</div>
             </div>
             <div className="text-center">
                <div className="text-red-400 text-[10px] uppercase font-bold tracking-wider">Excluded</div>
                <div className="text-red-500 text-2xl font-bold">-{summary.rejected || 0}</div>
             </div>
             <div className="text-center border-l border-gray-200 pl-6">
                <div className="text-green-600 text-[10px] uppercase font-bold tracking-wider">Kept</div>
                <div className="text-green-600 text-2xl font-bold">{summary.selected || 0}</div>
             </div>
        </div>

        {/* Filter Tabs */}
        <div className="flex gap-2 text-xs">
            <button onClick={() => setFilterMode('ALL')} className={`px-3 py-1 rounded-full border transition-colors ${filterMode==='ALL' ? 'bg-gray-800 text-white border-gray-800' : 'bg-white text-gray-500 hover:bg-gray-50'}`}>All Papers</button>
            <button onClick={() => setFilterMode('KEPT')} className={`px-3 py-1 rounded-full border transition-colors ${filterMode==='KEPT' ? 'bg-green-600 text-white border-green-600' : 'bg-white text-gray-500 hover:bg-green-50'}`}>Kept Only</button>
            <button onClick={() => setFilterMode('REJECTED')} className={`px-3 py-1 rounded-full border transition-colors ${filterMode==='REJECTED' ? 'bg-red-600 text-white border-red-600' : 'bg-white text-gray-500 hover:bg-red-50'}`}>Rejected Only</button>
        </div>

        {/* Paper List */}
        {papers.length > 0 ? (
            <div className="bg-gray-900 rounded-lg border border-gray-700 max-h-80 overflow-y-auto custom-scrollbar shadow-inner">
              <table className="w-full text-left text-xs">
                  <thead className="bg-gray-800 text-gray-400 sticky top-0 z-10">
                      <tr><th className="p-3 font-semibold">Paper Title</th><th className="p-3 text-right font-semibold">Status</th></tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                      {papers
                          .filter(p => {
                              const kept = isPaperKept(p.status);
                              if (filterMode === 'KEPT') return kept;
                              if (filterMode === 'REJECTED') return !kept;
                              return true;
                          })
                          .map((p, i) => {
                              const kept = isPaperKept(p.status);
                              return (
                                <tr key={i} className="hover:bg-gray-800/50 transition-colors">
                                    <td className="p-3 text-gray-300 truncate max-w-sm" title={p.title}>
                                        {p.title}
                                        {/* Optional: Show year if available */}
                                        {p.year && <span className="ml-2 text-[9px] text-gray-500 border border-gray-700 px-1 rounded">{p.year}</span>}
                                    </td>
                                    <td className="p-3 text-right">
                                        <span className={`px-2 py-1 rounded font-bold text-[10px] border shadow-sm ${
                                            kept 
                                            ? 'bg-green-600 text-white border-green-700' 
                                            : 'bg-red-600 text-white border-red-700'
                                        }`}>
                                            {p.status || "Kept"}
                                        </span>
                                    </td>
                                </tr>
                              );
                          })}
                  </tbody>
              </table>
            </div>
        ) : (
            <div className="text-center text-gray-400 py-8 italic border border-dashed border-gray-300 rounded-lg">
                No papers found yet.<br/>
                <span className="text-[10px] text-gray-500">Backend sent 0 items in 'data.papers'</span>
            </div>
        )}
    </div>
  );
};

export default SearchFilterView;