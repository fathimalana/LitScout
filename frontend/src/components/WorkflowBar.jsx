import React from 'react';

const WorkflowBar = ({ workflow, selectedStage, onSelectStage }) => {
    return (
        <div className="mb-8 animate-fadeIn">
            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 pl-1">Agent Workflow Pipeline</h3>
            <div className="flex space-x-3 overflow-x-auto pb-4 custom-scrollbar">
                {workflow.map((stage) => (
                    <button 
                        key={stage.id}
                        onClick={() => onSelectStage(stage)}
                        className={`flex-shrink-0 flex flex-col justify-between p-3 rounded-lg border transition-all w-40 h-28 text-left relative overflow-hidden group
                            ${selectedStage?.id === stage.id 
                                ? 'bg-blue-600 text-white border-blue-600 shadow-lg scale-105 z-10' 
                                : stage.status === 'idle' 
                                    ? 'bg-gray-50 border-gray-200 text-gray-400' 
                                    : 'bg-white border-gray-200 text-gray-600 hover:border-blue-300 hover:shadow-md'
                            }`}
                    >
                        {/* Running Pulse Indicator */}
                        {stage.status === 'running' && (
                            <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
                        )}

                        <div>
                            <div className={`text-[10px] uppercase font-bold tracking-wider mb-1 ${selectedStage?.id === stage.id ? 'text-blue-200' : 'text-gray-400'}`}>
                                {stage.role}
                            </div>
                            <div className="font-bold text-sm leading-tight">{stage.name}</div>
                        </div>
                        <div className={`text-[10px] mt-2 truncate ${selectedStage?.id === stage.id ? 'text-blue-100' : stage.status === 'completed' ? 'text-green-600 font-medium' : 'text-gray-400'}`}>
                            {stage.stats}
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );
};

export default WorkflowBar;