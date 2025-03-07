import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';

const FinAssistVisualizations = () => {
  // Feature importance data from risk model
  const featureImportanceData = [
    { feature: 'Age', importance: 0.22 },
    { feature: 'Income', importance: 0.18 },
    { feature: 'Debt to Income', importance: 0.15 },
    { feature: 'Credit Score', importance: 0.12 },
    { feature: 'Savings Rate', importance: 0.09 },
    { feature: 'Market Sentiment', importance: 0.07 },
    { feature: 'Employment Status', importance: 0.06 },
    { feature: 'Education Level', importance: 0.04 },
    { feature: 'Has Dependents', importance: 0.04 },
    { feature: 'Has Mortgage', importance: 0.03 },
  ];

  // Risk profiles distribution data
  const riskProfileData = [
    { name: 'Conservative', value: 35 },
    { name: 'Moderate', value: 45 },
    { name: 'Aggressive', value: 20 },
  ];

  // Model performance metrics
  const modelPerformanceData = [
    { metric: 'Accuracy', value: 0.87 },
    { metric: 'Precision', value: 0.85 },
    { metric: 'Recall', value: 0.82 },
    { metric: 'F1 Score', value: 0.84 },
  ];

  // Risk score distribution by age group
  const riskScoreByAgeData = [
    { age: '18-25', averageRiskScore: 75 },
    { age: '26-35', averageRiskScore: 68 },
    { age: '36-45', averageRiskScore: 58 },
    { age: '46-55', averageRiskScore: 49 },
    { age: '56-65', averageRiskScore: 35 },
    { age: '65+', averageRiskScore: 22 },
  ];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28'];

  return (
    <div className="p-4 space-y-8">
      <h1 className="text-2xl font-bold mb-8 text-center">FinAssist Data Visualizations</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Feature Importance Plot */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Feature Importance in Risk Assessment</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={featureImportanceData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 0.25]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
              <YAxis dataKey="feature" type="category" />
              <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
              <Bar dataKey="importance" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Profile Distribution */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Risk Profile Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskProfileData}
                cx="50%"
                cy="50%"
                labelLine={true}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              >
                {riskProfileData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => `${value}%`} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Model Performance Metrics */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Model Performance Metrics</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={modelPerformanceData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
              <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
              <Bar dataKey="value" fill="#00C49F" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Score by Age Group */}
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Average Risk Score by Age Group</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={riskScoreByAgeData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="age" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="averageRiskScore" 
                stroke="#FF8042" 
                name="Risk Score" 
                strokeWidth={2} 
                dot={{ r: 5 }} 
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="bg-gray-50 p-4 rounded-lg mt-6 text-center">
        <p className="text-sm text-gray-600">
          These visualizations demonstrate FinAssist's ability to provide transparent, 
          data-driven insights for financial advisors and customers.
        </p>
      </div>
    </div>
  );
};

export default FinAssistVisualizations;