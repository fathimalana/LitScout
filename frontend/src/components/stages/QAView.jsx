import React, { useState } from 'react';

const QAView = ({ data }) => {
    const [activeTab, setActiveTab] = useState('overview');

    const summary   = data?.summary || {};
    const covered   = data?.covered_questions || [];
    const missing   = data?.missing_questions || [];
    const suggestions = data?.suggestions || [];
    const reasoning = data?.reasoning || '';

    const score       = summary.score       ?? 0;
    const passed      = summary.passed      ?? false;
    const coverage    = summary.coverage_score ?? 0;
    const citations   = summary.citation_count ?? 0;
    const words       = summary.word_count  ?? 0;
    const confidence  = summary.confidence  ?? null;   // 0-1 float
    const disagreement = summary.disagreement ?? null; // score gap between LLMs
    const strengths   = summary.strengths   || [];
    const weaknesses  = summary.weaknesses  || [];

    // Colour helpers
    const scoreColour = score >= 7 ? '#16a34a' : score >= 5 ? '#d97706' : '#dc2626';
    const scoreBg     = score >= 7
        ? 'bg-green-50 border-green-200'
        : score >= 5
        ? 'bg-amber-50 border-amber-200'
        : 'bg-red-50 border-red-200';

    if (!summary.method && !reasoning && !score) {
        return (
            <div className="p-10 text-center">
                <div className="animate-spin h-8 w-8 border-4 border-violet-500 border-t-transparent rounded-full mx-auto mb-4" />
                <p className="text-gray-400 italic">Running quality checks…</p>
            </div>
        );
    }

    // Arc gauge for score
    const radius = 40;
    const circ   = 2 * Math.PI * radius;
    const arc    = circ * 0.75; // 270° sweep
    const fill   = arc * (score / 10);

    const tabs = [
        { id: 'overview',    label: 'Overview',    icon: '🧠' },
        { id: 'suggestions', label: 'Suggestions', icon: '💡', count: suggestions.length },
    ];

    return (
        <div className="space-y-4 animate-fadeIn">

            {/* ── Hero Score Row ── */}
            <div className={`rounded-2xl border-2 p-5 flex items-center gap-6 ${scoreBg}`}>
                {/* Arc gauge */}
                <div className="relative flex-shrink-0 w-24 h-24">
                    <svg viewBox="0 0 100 100" className="w-full h-full -rotate-[135deg]">
                        {/* Track */}
                        <circle cx="50" cy="50" r={radius} fill="none"
                            stroke="#e5e7eb" strokeWidth="10"
                            strokeDasharray={`${arc} ${circ}`}
                            strokeLinecap="round" />
                        {/* Fill */}
                        <circle cx="50" cy="50" r={radius} fill="none"
                            stroke={scoreColour} strokeWidth="10"
                            strokeDasharray={`${fill} ${circ}`}
                            strokeLinecap="round"
                            style={{ transition: 'stroke-dasharray 0.8s ease' }} />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center rotate-0">
                        <span className="text-2xl font-black leading-none" style={{ color: scoreColour }}>
                            {typeof score === 'number' ? score.toFixed(1) : score}
                        </span>
                        <span className="text-[10px] text-gray-400 font-bold">/ 10</span>
                    </div>
                </div>

                {/* Verdict + meta */}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2 flex-wrap">
                        <span className={`text-sm font-extrabold px-4 py-1.5 rounded-full ${passed ? 'bg-green-500 text-white' : 'bg-amber-400 text-white'}`}>
                            {passed ? '✓ PASSED' : '⚠ NEEDS WORK'}
                        </span>
                        {confidence !== null && (
                            <span className="text-xs font-semibold bg-white/80 border border-gray-200 px-3 py-1 rounded-full text-gray-600">
                                🤝 Confidence {Math.round(confidence * 100)}%
                            </span>
                        )}
                        {disagreement !== null && (
                            <span className={`text-xs font-semibold px-3 py-1 rounded-full ${disagreement > 2 ? 'bg-orange-100 text-orange-600' : 'bg-teal-100 text-teal-600'}`}>
                                ↕ LLM Gap {disagreement.toFixed(1)} pts
                            </span>
                        )}
                    </div>
                    <p className="text-xs text-gray-500 italic">
                        Evaluated by two LLMs (Llama‑70B + Llama‑8B) using confidence-weighted averaging
                    </p>
                </div>
            </div>

            {/* ── 4-stat grid ── */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                    { label: 'RQ Coverage',  value: `${Math.round(coverage * 100)}%`, bg: 'bg-blue-50 border-blue-100',     text: 'text-blue-700',   sub: 'text-blue-500' },
                    { label: 'Citations',    value: citations,                         bg: 'bg-violet-50 border-violet-100', text: 'text-violet-700', sub: 'text-violet-500' },
                    { label: 'Word Count',   value: words.toLocaleString(),            bg: 'bg-slate-50 border-slate-100',   text: 'text-slate-700',  sub: 'text-slate-500' },
                    { label: 'RQs Covered',  value: `${covered.length}/${covered.length + missing.length}`, bg: 'bg-emerald-50 border-emerald-100', text: 'text-emerald-700', sub: 'text-emerald-500' },
                ].map(({ label, value, bg, text, sub }) => (
                    <div key={label} className={`${bg} border rounded-xl p-3 text-center`}>
                        <p className={`text-[10px] font-bold uppercase tracking-wider mb-1 ${sub}`}>{label}</p>
                        <p className={`text-2xl font-black ${text}`}>{value}</p>
                    </div>
                ))}
            </div>

            {/* ── Tab Bar ── */}
            <div className="flex gap-1 bg-gray-100 p-1 rounded-xl overflow-x-auto">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex-shrink-0 flex items-center gap-1.5 text-xs font-bold px-3 py-1.5 rounded-lg transition-all ${
                            activeTab === tab.id
                                ? 'bg-white text-gray-900 shadow-sm'
                                : 'text-gray-500 hover:text-gray-700'
                        }`}
                    >
                        <span>{tab.icon}</span>
                        <span>{tab.label}</span>
                        {tab.count !== undefined && tab.count > 0 && (
                            <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-bold ml-0.5 ${
                                activeTab === tab.id ? 'bg-gray-100 text-gray-600' : 'bg-gray-200 text-gray-500'
                            }`}>
                                {tab.count}
                            </span>
                        )}
                    </button>
                ))}
            </div>

            {/* ── Tab Content ── */}

            {activeTab === 'overview' && (
                <div className="bg-white border border-gray-200 rounded-xl p-5 space-y-4">
                    <div>
                        <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">Reviewer Reasoning</p>
                        <p className="text-sm text-slate-700 leading-relaxed font-serif">
                            {reasoning || 'Multi-LLM evaluation — two models independently scored this draft and their results were combined using confidence-weighted averaging.'}
                        </p>
                    </div>
                    {/* Threshold badges */}
                    <div className="border-t pt-3 grid grid-cols-2 gap-2 text-xs">
                        {[
                            { label: 'Score ≥ 5.0',    ok: score >= 5 },
                            { label: 'Citations ≥ 1',  ok: citations >= 1 },
                            { label: 'Words ≥ 100',    ok: words >= 100 },
                            { label: 'LLM Gap < 3.0',  ok: disagreement === null || disagreement < 3 },
                        ].map(({ label, ok }) => (
                            <div key={label} className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'}`}>
                                <span>{ok ? '✓' : '✗'}</span>
                                <span className="font-semibold">{label}</span>
                            </div>
                        ))}
                    </div>
                    <div className="flex gap-2 flex-wrap">
                        <span className="text-[10px] bg-slate-100 text-slate-500 px-2 py-1 rounded-full font-bold uppercase tracking-wider">
                            {summary.method || 'LangGraph QA pipeline'}
                        </span>
                        <span className={`text-[10px] px-2 py-1 rounded-full font-bold uppercase tracking-wider ${
                            summary.status === 'passed' ? 'bg-green-100 text-green-600' : 'bg-amber-100 text-amber-600'
                        }`}>
                            {summary.status || 'unknown'}
                        </span>
                    </div>
                </div>
            )}

{activeTab === 'suggestions' && (
                <div className="bg-white border border-gray-200 rounded-xl p-5">
                    <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3">Improvement Suggestions</p>
                    {suggestions.length > 0 ? (
                        <ol className="space-y-2">
                            {suggestions.map((s, i) => (
                                <li key={i} className="flex gap-3 text-sm text-slate-700 bg-slate-50 border border-slate-100 rounded-lg px-4 py-3">
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
