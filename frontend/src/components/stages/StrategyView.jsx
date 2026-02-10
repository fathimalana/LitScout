import React from 'react';

const StrategyView = ({ data }) => {
  return (
    <div className="space-y-4 animate-fadeIn">
      <div>
        <span className="text-blue-600 font-bold block mb-2 text-xs uppercase tracking-wider">Research Questions</span>
        <ul className="list-disc pl-5 text-gray-700 text-sm space-y-1">
          {data.questions && data.questions.map((q, i) => <li key={i}>{q}</li>)}
        </ul>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-green-50 p-4 rounded-lg border border-green-100">
          <span className="text-green-700 font-bold block text-xs mb-2 uppercase">Inclusion Criteria</span>
          <p className="text-gray-600 text-xs leading-relaxed">{data.criteria?.inclusion}</p>
        </div>
        <div className="bg-red-50 p-4 rounded-lg border border-red-100">
          <span className="text-red-700 font-bold block text-xs mb-2 uppercase">Exclusion Criteria</span>
          <p className="text-gray-600 text-xs leading-relaxed">{data.criteria?.exclusion}</p>
        </div>
      </div>
    </div>
  );
};

export default StrategyView;