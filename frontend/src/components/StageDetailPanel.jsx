import React from 'react';
import StrategyView from './stages/StrategyView';
import SearchFilterView from './stages/SearchFilterView';
import ScreeningView from './stages/ScreeningView';
import ExtractionView from './stages/ExtractionView';
import ThematicAnalysisView from './stages/ThematicAnalysisView';
import SynthesisView from './stages/SynthesisView';
import QAView from './stages/QAView';
import ReportView from './stages/ReportView';

const StageDetailPanel = ({ stage, onClose }) => {
  if (!stage) return null;

  const supportedRoles = [
    "Strategy",
    "Search & Filter",
    "Screening",
    "Extraction",
    "PDF Processing",
    "Thematic Analysis",
    "Synthesis",
    "Quality Assurance",
    "Report Generation"
  ];

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
        {stage.status === 'idle' && (
          <div className="text-gray-400 italic text-center p-10 border-2 border-dashed border-gray-100 rounded-xl m-4">
            Waiting for research to start... <br />
            Enter a topic above to activate this agent.
          </div>
        )}

        {stage.status !== 'idle' && (
          <>
            {/* Stage 1 */}
            {stage.role === "Strategy" && <StrategyView data={stage.data} />}

            {/* Stage 2 */}
            {stage.role === "Search & Filter" && <SearchFilterView data={stage.data} />}

            {/* Stage 3 */}
            {stage.role === "Screening" && <ScreeningView data={stage.data} />}

            {/* Stage 4 */}
            {(stage.role === "Extraction" || stage.role === "PDF Processing") && (
              <ExtractionView data={stage.data} />
            )}

            {/* Stage 5 */}
            {stage.role === "Thematic Analysis" && (
              <ThematicAnalysisView data={stage.data} />
            )}

            {/* Stage 6 */}
            {stage.role === "Synthesis" && (
              <SynthesisView data={stage.data} />
            )}

            {/* Stage 7 */}
            {stage.role === "Quality Assurance" && (
              <QAView data={stage.data} />
            )}

            {/* Stage 8 */}
            {stage.role === "Report Generation" && (
              <ReportView data={stage.data} />
            )}

            {/* Fallback */}
            {!supportedRoles.includes(stage.role) && (
              <div className="text-gray-500 italic text-center p-8">
                The detailed view for <strong>{stage.name}</strong> is under construction. <br />
                <div className="mt-4 p-3 bg-gray-50 rounded border border-gray-100 text-[10px] font-mono text-left overflow-hidden">
                  <span className="text-blue-600 font-bold">Detected Role:</span> {stage.role} <br />
                  <span className="text-emerald-600 font-bold">Raw data:</span> {JSON.stringify(stage.data).slice(0, 100)}...
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default StageDetailPanel;