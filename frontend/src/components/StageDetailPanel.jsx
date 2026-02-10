import React from 'react';
import StrategyView from './stages/StrategyView';
import SearchFilterView from './stages/SearchFilterView';
import ScreeningView from './stages/ScreeningView';

const StageDetailPanel = ({ stage, onClose }) => {
  if (!stage) return null;

  return (
    <div className="bg-white rounded-xl shadow-xl mb-6 animate-slideUp border border-gray-200 flex flex-col h-[500px] overflow-hidden">
      
      {/* Header */}
      <div className="bg-gray-50 border-b border-gray-100 p-4 flex justify-between items-center flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${stage.status === 'running' ? 'bg-blue-500 animate-pulse' : stage.status === 'completed' ? 'bg-green-500' : 'bg-gray-300'}`}></span>
            <h3 className="text-gray-700 font-bold text-sm uppercase tracking-wide">Output: {stage.name}</h3>
          </div>
          <button onClick={onClose} className="text-xs text-gray-400 hover:text-red-500 font-bold transition-colors uppercase tracking-wider">✕ Close Panel</button>
      </div>
      
      {/* Content Area */}
      <div className="flex-grow overflow-y-auto p-6 custom-scrollbar">
          
          {/* Idle State */}
          {stage.status === 'idle' && (
              <div className="text-gray-400 italic text-center p-10 border-2 border-dashed border-gray-100 rounded-xl m-4">
                  Waiting for research to start... <br/>
                  Enter a topic above to activate this agent.
              </div>
          )}

          {/* Render specific views based on Role */}
          {stage.status !== 'idle' && (
             <>
                {stage.role === "Strategy" && <StrategyView data={stage.data} />}
                
                {stage.role === "Search & Filter" && <SearchFilterView data={stage.data} />}
                
                {stage.role === "Screening" && <ScreeningView data={stage.data} />}
                
                {/* Fallback for other stages (Extraction, Synthesis, etc.) */}
                {!['Strategy', 'Search & Filter', 'Screening'].includes(stage.role) && (
                    <div className="text-gray-500 italic text-center p-8">
                        The detailed view for <strong>{stage.name}</strong> is under construction. <br/>
                        <span className="text-xs text-gray-400">Raw data: {JSON.stringify(stage.data).slice(0, 50)}...</span>
                    </div>
                )}
             </>
          )}
      </div>
    </div>
  );
};

export default StageDetailPanel;