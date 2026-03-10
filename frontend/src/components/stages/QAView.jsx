import React, { useState } from 'react';

const QAView = ({ data }) => {
    const [activeTab, setActiveTab] = useState('overview');

    const summary = data?.summary || {};
    const covered = data?.covered_questions || [];
    const missing = data?.missing_questions || [];
    const suggestions = data?.suggestions || [];
    const reasoning = data?.reasoning || '';

    const score = summary.score ?? 0;
    const passed = summary.passed ?? false;
    const coverage = summary.coverage_score ?? 0;
    const citations = summary.citation_count ?? 0;
    const words = summary.word_count ?? 0;

    // Colour helper for quality score
    const scoreColour = score >= 7 ? 'text-green-600' : score >= 5 ? 'text-amber-500' : 'text-red-500';
    const scoreBg = score >= 7 ? 'bg-green-50 border-green-100' : score >= 5 ? 'bg-amber-50 border-amber-100' : 'bg-red-50 border-red-100';

    if (!summary.method && !reasoning) {
        return (
            <div className="p-10 text-center text-gray-400 italic">
                Running quality checks...
            </div>
        );
    }

    return (
        <div className="space-y-5 animate-fadeIn">

            {/* Score Banner */}
            <div className={`rounded-xl border p-4 flex items-center justify-between ${scoreBg}`}>
                <div>
                    <p className="text-xs font-bold uppercase tracking-wider text-gray-500">Quality Score</p>
                    <p className={`text-4xl font-black ${scoreColour}`}>
                        {typeof score === 'number' ? score.toFixed(1) : score}
                        <span className="text-lg font-medium text-gray-400">/10</span>
                    </p>
                </div>
                <div className={`text-sm font-bold px-4 py-2 rounded-full ${passed ? 'bg-green-500 text-white' : 'bg-amber-400 text-white'}`}>
                    {passed ? '✓ PASSED' : '⚠ NEEDS WORK'}
                </div>
            </div>

            {/* Stats Row */}
            <div className="grid grid-cols-3 gap-3">
                <div className="bg-blue-50 p-3 rounded-xl border border-blue-100 text-center">
                    <p className="text-[10px] font-bold text-blue-500 uppercase tracking-wider">RQ Coverage</p>
                    <p className="text-2xl font-black text-blue-700">
                        {typeof coverage === 'number' ? `${Math.round(coverage * 100)}%` : coverage}
                    </p>
                </div>
                <div className="bg-purple-50 p-3 rounded-xl border border-purple-100 text-center">
                    <p className="text-[10px] font-bold text-purple-500 uppercase tracking-wider">Citations</p>
                    <p className="text-2xl font-black text-purple-700">{citations}</p>
                </div>
                <div className="bg-slate-50 p-3 rounded-xl border border-slate-100 text-center">
                    <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Words</p>
                    <p className="text-2xl font-black text-slate-700">{words}</p>
                </div>
            </div>

            {/* Tab Navigation */}
            <div className="flex gap-1 bg-gray-100 p-1 rounded-lg">
                {['overview', 'coverage', 'suggestions'].map(tab => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`flex-1 text-xs font-bold py-1.5 rounded-md capitalize transition-colors ${activeTab === tab
                                ? 'bg-white text-gray-800 shadow-sm'
                                : 'text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        {tab}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            {activeTab === 'overview' && (
                <div className="bg-white border border-gray-200 rounded-xl p-5 space-y-3">
                    <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Reviewer Reasoning</p>
                    <p className="text-sm text-slate-700 leading-relaxed font-serif">
                        {reasoning || 'No reasoning provided.'}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                        <span className="text-[10px] bg-slate-100 text-slate-500 px-2 py-1 rounded-full font-bold uppercase tracking-wider">
                            {summary.method || 'LangGraph QA pipeline'}
                        </span>
                        <span className={`text-[10px] px-2 py-1 rounded-full font-bold uppercase tracking-wider ${summary.status === 'passed' ? 'bg-green-100 text-green-600' : 'bg-amber-100 text-amber-600'
                            }`}>
                            {summary.status || 'unknown'}
                        </span>
                    </div>
                </div>
            )}

            {activeTab === 'coverage' && (
                <div className="space-y-3">
                    {covered.length > 0 && (
                        <div className="bg-green-50 border border-green-100 rounded-xl p-4">
                            <p className="text-[10px] font-bold text-green-600 uppercase tracking-wider mb-2">
                                ✓ Covered Questions ({covered.length})
                            </p>
                            <ul className="space-y-1">
                                {covered.map((q, i) => (
                                    <li key={i} className="text-sm text-green-800 flex gap-2">
                                        <span className="text-green-400 mt-0.5 flex-shrink-0">✓</span>
                                        {q}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                    {missing.length > 0 && (
                        <div className="bg-red-50 border border-red-100 rounded-xl p-4">
                            <p className="text-[10px] font-bold text-red-500 uppercase tracking-wider mb-2">
                                ✗ Missing Questions ({missing.length})
                            </p>
                            <ul className="space-y-1">
                                {missing.map((q, i) => (
                                    <li key={i} className="text-sm text-red-700 flex gap-2">
                                        <span className="text-red-400 mt-0.5 flex-shrink-0">✗</span>
                                        {q}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                    {covered.length === 0 && missing.length === 0 && (
                        <p className="text-gray-400 italic text-center py-6">No coverage data available.</p>
                    )}
                </div>
            )}

            {activeTab === 'suggestions' && (
                <div className="bg-white border border-gray-200 rounded-xl p-5">
                    <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3">
                        Improvement Suggestions
                    </p>
                    {suggestions.length > 0 ? (
                        <ol className="space-y-2">
                            {suggestions.map((s, i) => (
                                <li key={i} className="flex gap-3 text-sm text-slate-700">
                                    <span className="flex-shrink-0 w-5 h-5 bg-amber-100 text-amber-600 rounded-full text-xs flex items-center justify-center font-bold">
                                        {i + 1}
                                    </span>
                                    {s}
                                </li>
                            ))}
                        </ol>
                    ) : (
                        <p className="text-gray-400 italic text-center py-4">No suggestions — great quality!</p>
                    )}
                </div>
            )}

        </div>
    );
};

export default QAView;
