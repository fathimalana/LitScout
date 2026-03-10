import React, { useState } from 'react';

const SynthesisView = ({ data }) => {
    const [copied, setCopied] = useState(false);

    const summary = data?.summary || {};
    const draft = data?.draft || '';

    if (!draft) {
        return (
            <div className="p-10 text-center text-gray-400 italic">
                Generating synthesis draft...
            </div>
        );
    }

    const handleCopy = () => {
        navigator.clipboard.writeText(draft).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        });
    };

    // Split prose into paragraphs for clean rendering
    const paragraphs = draft.split(/\n+/).filter(p => p.trim().length > 0);

    return (
        <div className="space-y-6 animate-fadeIn">

            {/* Stats Row */}
            <div className="grid grid-cols-3 gap-4">
                <div className="bg-emerald-50 p-4 rounded-xl border border-emerald-100">
                    <p className="text-[10px] font-bold text-emerald-500 uppercase tracking-wider">Word Count</p>
                    <p className="text-2xl font-black text-emerald-700">{summary.word_count ?? paragraphs.join(' ').split(' ').length}</p>
                </div>
                <div className="bg-blue-50 p-4 rounded-xl border border-blue-100">
                    <p className="text-[10px] font-bold text-blue-500 uppercase tracking-wider">Themes Covered</p>
                    <p className="text-2xl font-black text-blue-700">{summary.total_themes ?? '—'}</p>
                </div>
                <div className="bg-purple-50 p-4 rounded-xl border border-purple-100">
                    <p className="text-[10px] font-bold text-purple-500 uppercase tracking-wider">Sections</p>
                    <p className="text-2xl font-black text-purple-700">{summary.section_count ?? '—'}</p>
                </div>
            </div>

            {/* Method badge */}
            {summary.method && (
                <div className="flex items-center gap-2">
                    <span className="text-[10px] bg-slate-100 text-slate-500 px-2 py-1 rounded-full font-bold uppercase tracking-wider">
                        {summary.method}
                    </span>
                    <span className={`text-[10px] px-2 py-1 rounded-full font-bold uppercase tracking-wider ${summary.status === 'complete'
                            ? 'bg-green-100 text-green-600'
                            : 'bg-amber-100 text-amber-600'
                        }`}>
                        {summary.status ?? 'complete'}
                    </span>
                </div>
            )}

            {/* Synthesis Draft */}
            <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
                {/* Draft header */}
                <div className="flex justify-between items-center px-5 py-3 bg-gray-50 border-b border-gray-100">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-widest flex items-center gap-2">
                        <span className="text-emerald-500">✦</span> Synthesis Draft
                    </span>
                    <button
                        onClick={handleCopy}
                        className="text-[11px] text-gray-400 hover:text-blue-600 transition-colors font-medium flex items-center gap-1"
                    >
                        {copied ? '✓ Copied!' : '⎘ Copy'}
                    </button>
                </div>

                {/* Prose body */}
                <div className="p-6 space-y-4 text-sm text-slate-700 leading-relaxed font-serif max-h-[320px] overflow-y-auto custom-scrollbar">
                    {paragraphs.map((para, idx) => (
                        <p key={idx} className={
                            idx === 0
                                ? 'text-slate-800 font-medium'             // intro paragraph bolder
                                : idx === paragraphs.length - 1
                                    ? 'text-slate-600 italic'                // conclusion slightly muted
                                    : ''
                        }>
                            {para}
                        </p>
                    ))}
                </div>
            </div>

        </div>
    );
};

export default SynthesisView;
