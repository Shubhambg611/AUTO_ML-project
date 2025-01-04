import React from 'react';
import { Bar, Line } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

const ModelResultsDashboard = ({ results }) => {
  // Metrics Summary Cards
  const MetricCard = ({ title, value, trend, trendValue }) => (
    <div className="bg-gradient-to-br from-gray-700 to-gray-800 rounded-lg p-6 text-white">
      <h3 className="text-4xl font-bold mb-2">{value}</h3>
      <p className="text-sm opacity-80">{title}</p>
      <div className="mt-2 text-sm">
        <span className={`${trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
          {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}%
        </span>
        <span className="ml-2">vs previous run</span>
      </div>
    </div>
  );

  // Performance Gauges
  const PerformanceGauge = ({ title, value, previousValue }) => {
    const trend = ((value - previousValue) / previousValue * 100).toFixed(2);
    return (
      <div className="bg-white rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">{title}</h3>
        <div className="relative pt-1">
          <div className="flex mb-2 items-center justify-between">
            <div>
              <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-blue-600 bg-blue-200">
                Current: {value}%
              </span>
            </div>
            <div className="text-right">
              <span className={`text-xs font-semibold inline-block px-2 py-1 rounded-full ${trend > 0 ? 'text-green-600 bg-green-200' : 'text-red-600 bg-red-200'}`}>
                {trend}%
              </span>
            </div>
          </div>
          <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-blue-200">
            <div 
              className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500"
              style={{ width: `${value}%` }}
            ></div>
          </div>
        </div>
      </div>
    );
  };

  // Model Comparison Chart
  const ModelComparisonChart = ({ models }) => (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Model Performance Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <Bar
            data={models}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="accuracy" fill="#8884d8" />
            <Bar dataKey="precision" fill="#82ca9d" />
            <Bar dataKey="recall" fill="#ffc658" />
          </Bar>
        </div>
      </CardContent>
    </Card>
  );

  // Performance Timeline
  const PerformanceTimeline = ({ data }) => (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Performance Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <Line
            data={data}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="accuracy" stroke="#8884d8" />
            <Line type="monotone" dataKey="precision" stroke="#82ca9d" />
          </Line>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="p-6 space-y-6">
      {/* Top Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <MetricCard 
          title="Overall Accuracy"
          value={`${(results.best_model.accuracy * 100).toFixed(1)}%`}
          trend={2.5}
          trendValue="vs previous"
        />
        <MetricCard 
          title="Total Features"
          value={results.feature_count}
          trend={0}
          trendValue="no change"
        />
        <MetricCard 
          title="Training Time"
          value={`${results.training_time}s`}
          trend={-15}
          trendValue="faster"
        />
        <MetricCard 
          title="Cross-Val Score"
          value={`${(results.cv_score * 100).toFixed(1)}%`}
          trend={1.2}
          trendValue="improved"
        />
        <MetricCard 
          title="F1 Score"
          value={`${(results.best_model.f1 * 100).toFixed(1)}%`}
          trend={3.1}
          trendValue="improved"
        />
      </div>

      {/* Performance Gauges */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PerformanceGauge 
          title="Model Training Accuracy"
          value={95.5}
          previousValue={92.3}
        />
        <PerformanceGauge 
          title="Validation Accuracy"
          value={93.2}
          previousValue={91.8}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ModelComparisonChart models={results.model_comparison} />
        <PerformanceTimeline data={results.performance_timeline} />
      </div>
    </div>
  );
};

export default ModelResultsDashboard;