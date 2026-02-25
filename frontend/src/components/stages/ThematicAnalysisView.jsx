import React from 'react';

const ThematicAnalysisView = ({ data }) => {
  const themes = data?.themes || [];

  if (themes.length === 0) {
    return <div className="p-10 text-center text-gray-400">Identifying common themes across papers...</div>;
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-purple-50 p-4 rounded-xl border border-purple-100">
          <p className="text-[10px] font-bold text-purple-500 uppercase">Total Themes</p>
          <p className="text-2xl font-black text-purple-700">{themes.length}</p>
        </div>
      </div>

      <div className="space-y-4">
        {themes.map((theme, idx) => (
          <div key={idx} className="bg-white border border-gray-100 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex justify-between items-start mb-2">
              <h4 className="font-bold text-slate-800 text-lg flex items-center gap-2">
                <span className="bg-purple-100 text-purple-600 px-2 py-0.5 rounded text-xs">Theme {idx + 1}</span>
                {theme.name}
              </h4>
              <span className="text-[10px] bg-slate-100 text-slate-500 px-2 py-1 rounded-full font-bold">
                {theme.papers_count} PAPERS
              </span>
            </div>
            <p className="text-sm text-slate-600 mb-4 italic">{theme.description}</p>
            
            <div className="space-y-2">
              {theme.findings?.map((finding, fIdx) => (
                <div key={fIdx} className="flex gap-2 text-sm text-slate-700 bg-slate-50 p-2 rounded">
                  <span className="text-purple-400">•</span>
                  {finding}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ThematicAnalysisView;