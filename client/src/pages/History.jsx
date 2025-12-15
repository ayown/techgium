import React, { useState } from 'react';

/**
 * History Component
 * - Displays user's diagnostic history and past assessments
 * - Filterable and searchable interface
 * - Shows assessment details and results
 */
const History = () => {
  // Sample data - replace with actual API data
  const [assessments] = useState([
    {
      id: 1,
      date: '2024-01-15',
      type: 'General Health Assessment',
      status: 'Completed',
      score: 85,
      symptoms: ['Headache', 'Fatigue'],
      duration: '15 minutes'
    },
    {
      id: 2,
      date: '2024-01-10',
      type: 'Eye Examination',
      status: 'Completed',
      score: 92,
      symptoms: ['Eye strain', 'Blurred vision'],
      duration: '12 minutes'
    },
    {
      id: 3,
      date: '2024-01-05',
      type: 'Skin Analysis',
      status: 'Pending Review',
      score: null,
      symptoms: ['Rash', 'Itching'],
      duration: '8 minutes'
    },
    {
      id: 4,
      date: '2024-01-01',
      type: 'Lab Report Analysis',
      status: 'Completed',
      score: 78,
      symptoms: ['Blood pressure concerns'],
      duration: '20 minutes'
    }
  ]);

  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('All');

  /**
   * Filter assessments based on search term and status
   */
  const filteredAssessments = assessments.filter(assessment => {
    const matchesSearch = assessment.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         assessment.symptoms.some(symptom => 
                           symptom.toLowerCase().includes(searchTerm.toLowerCase())
                         );
    const matchesStatus = filterStatus === 'All' || assessment.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

  /**
   * Get status badge styling
   */
  const getStatusBadge = (status) => {
    const baseClasses = "px-2 py-1 text-xs font-medium rounded-full";
    switch (status) {
      case 'Completed':
        return `${baseClasses} bg-green-100 text-green-800`;
      case 'Pending Review':
        return `${baseClasses} bg-yellow-100 text-yellow-800`;
      case 'In Progress':
        return `${baseClasses} bg-blue-100 text-blue-800`;
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`;
    }
  };

  /**
   * Get health score color
   */
  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Assessment History</h1>
          <p className="mt-1 text-sm text-gray-600">
            View your past health assessments and track your progress.
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors duration-200">
            Export History
          </button>
        </div>
      </div>

      {/* Search and filter controls */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0 sm:space-x-4">
          
          {/* Search input */}
          <div className="flex-1 max-w-md">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                type="text"
                placeholder="Search assessments..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>

          {/* Status filter */}
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">Filter by status:</label>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="All">All</option>
              <option value="Completed">Completed</option>
              <option value="Pending Review">Pending Review</option>
              <option value="In Progress">In Progress</option>
            </select>
          </div>
        </div>
      </div>

      {/* Assessment cards */}
      <div className="space-y-4">
        {filteredAssessments.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-gray-900">No assessments found</h3>
            <p className="mt-1 text-sm text-gray-500">
              {searchTerm || filterStatus !== 'All' 
                ? 'Try adjusting your search or filter criteria.'
                : 'Start your first health assessment to see your history here.'
              }
            </p>
          </div>
        ) : (
          filteredAssessments.map((assessment) => (
            <div key={assessment.id} className="bg-white rounded-lg shadow hover:shadow-md transition-shadow duration-200">
              <div className="p-6">
                <div className="flex items-start justify-between">
                  
                  {/* Assessment info */}
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <h3 className="text-lg font-medium text-gray-900">{assessment.type}</h3>
                      <span className={getStatusBadge(assessment.status)}>
                        {assessment.status}
                      </span>
                    </div>
                    
                    <div className="mt-2 flex items-center space-x-6 text-sm text-gray-500">
                      <div className="flex items-center space-x-1">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3a2 2 0 012-2h4a2 2 0 012 2v4m-6 4v10a2 2 0 002 2h4a2 2 0 002-2V11m-6 0V9a2 2 0 012-2h4a2 2 0 012 2v2m-6 0h8" />
                        </svg>
                        <span>{new Date(assessment.date).toLocaleDateString()}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>{assessment.duration}</span>
                      </div>
                    </div>

                    {/* Symptoms */}
                    <div className="mt-3">
                      <p className="text-sm text-gray-600 mb-2">Symptoms assessed:</p>
                      <div className="flex flex-wrap gap-2">
                        {assessment.symptoms.map((symptom, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full"
                          >
                            {symptom}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Health score and actions */}
                  <div className="flex flex-col items-end space-y-3">
                    {assessment.score && (
                      <div className="text-center">
                        <p className="text-sm text-gray-500">Health Score</p>
                        <p className={`text-2xl font-bold ${getScoreColor(assessment.score)}`}>
                          {assessment.score}/100
                        </p>
                      </div>
                    )}
                    
                    <div className="flex space-x-2">
                      <button className="px-3 py-1 text-sm text-blue-600 border border-blue-600 rounded-md hover:bg-blue-50 transition-colors duration-200">
                        View Details
                      </button>
                      {assessment.status === 'Completed' && (
                        <button className="px-3 py-1 text-sm text-green-600 border border-green-600 rounded-md hover:bg-green-50 transition-colors duration-200">
                          Download Report
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Summary stats */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Assessment Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">{assessments.length}</p>
            <p className="text-sm text-gray-600">Total Assessments</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">
              {assessments.filter(a => a.status === 'Completed').length}
            </p>
            <p className="text-sm text-gray-600">Completed</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-yellow-600">
              {assessments.filter(a => a.score && a.score >= 80).length}
            </p>
            <p className="text-sm text-gray-600">High Scores (80+)</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default History;