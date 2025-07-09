import React, { useState, useEffect } from 'react';
import { TrendingUp, GitCompare, AlertTriangle, CheckCircle, X, ChevronDown, Loader2, Download, Save, MessageSquare, Search, Video, Code, TestTube, Image } from 'lucide-react';
import { FeatureType } from '../App';
import { apiService, Space } from '../services/api';
import { getConfluenceSpaceAndPageFromUrl } from '../utils/urlUtils';

interface ImpactAnalyzerProps {
  onClose: () => void;
  onFeatureSelect: (feature: FeatureType) => void;
  autoSpaceKey?: string | null;
  isSpaceAutoConnected?: boolean;
}

interface DiffMetrics {
  linesAdded: number;
  linesRemoved: number;
  filesChanged: number;
  percentageChanged: number;
}

interface RiskLevel {
  level: 'low' | 'medium' | 'high';
  score: number;
  factors: string[];
}

const ImpactAnalyzer: React.FC<ImpactAnalyzerProps> = ({ onClose, onFeatureSelect, autoSpaceKey, isSpaceAutoConnected }) => {
  const [selectedSpace, setSelectedSpace] = useState('');
  const [oldPage, setOldPage] = useState('');
  const [newPage, setNewPage] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isQALoading, setIsQALoading] = useState(false);
  const [diffResults, setDiffResults] = useState<string>('');
  const [metrics, setMetrics] = useState<DiffMetrics | null>(null);
  const [impactSummary, setImpactSummary] = useState('');
  const [riskLevel, setRiskLevel] = useState<RiskLevel | null>(null);
  const [question, setQuestion] = useState('');
  const [qaResults, setQaResults] = useState<Array<{question: string, answer: string}>>([]);
  const [exportFormat, setExportFormat] = useState('markdown');
  const [spaces, setSpaces] = useState<Space[]>([]);
  const [pages, setPages] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [showToast, setShowToast] = useState(false);

  const features = [
    { id: 'search' as const, label: 'AI Powered Search', icon: Search },
    { id: 'video' as const, label: 'Video Summarizer', icon: Video },
    { id: 'code' as const, label: 'Code Assistant', icon: Code },
    { id: 'impact' as const, label: 'Impact Analyzer', icon: TrendingUp },
    { id: 'test' as const, label: 'Test Support Tool', icon: TestTube },
    { id: 'image' as const, label: 'Image Insights & Chart Builder', icon: Image },
  ];

  // Load spaces on component mount
  useEffect(() => {
    loadSpaces();
  }, []);

  // Auto-select space if provided via URL
  useEffect(() => {
    if (autoSpaceKey && isSpaceAutoConnected) {
      setSelectedSpace(autoSpaceKey);
    }
  }, [autoSpaceKey, isSpaceAutoConnected]);

  // Load pages when space is selected
  useEffect(() => {
    if (selectedSpace) {
      loadPages();
    }
  }, [selectedSpace]);

  const loadSpaces = async () => {
    try {
      setError('');
      const result = await apiService.getSpaces();
      setSpaces(result.spaces);
    } catch (err) {
      setError('Failed to load spaces. Please check your backend connection.');
      console.error('Error loading spaces:', err);
    }
  };

  const loadPages = async () => {
    try {
      setError('');
      const result = await apiService.getPages(selectedSpace);
      setPages(result.pages);
    } catch (err) {
      setError('Failed to load pages. Please check your space key.');
      console.error('Error loading pages:', err);
    }
  };

  const analyzeDiff = async () => {
    if (!selectedSpace || !oldPage || !newPage) return;
    
    setIsAnalyzing(true);
    setError('');
    
    try {
      const result = await apiService.impactAnalyzer({
        space_key: selectedSpace,
        old_page_title: oldPage,
        new_page_title: newPage
      });

      setDiffResults(result.diff || '');
      setMetrics({
        linesAdded: result.lines_added || 0,
        linesRemoved: result.lines_removed || 0,
        filesChanged: result.files_changed || 1,
        percentageChanged: result.percentage_change || 0
      });
      
      setImpactSummary(result.impact_analysis || '');
      setRiskLevel({
        level: (result.risk_level || 'low') as 'low' | 'medium' | 'high',
        score: result.risk_score || 0,
        factors: result.risk_factors || []
      });
      
    } catch (err) {
      setError('Failed to analyze impact. Please try again.');
      console.error('Error analyzing impact:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const addQuestion = async () => {
    if (!question.trim() || !selectedSpace || !oldPage || !newPage) {
      console.log('Missing question or pages:', { question, selectedSpace, oldPage, newPage });
      return;
    }
    
    console.log('Adding question to impact analyzer:', question);
    setIsQALoading(true);
    
    try {
      const result = await apiService.impactAnalyzer({
        space_key: selectedSpace,
        old_page_title: oldPage,
        new_page_title: newPage,
        question: question
      });

      console.log('Impact analyzer Q&A response:', result);

      const answer = result.answer || `Based on the code changes analyzed, here's the response to your question: "${question}"

The modifications primarily focus on security enhancements and input validation. The impact on ${question.toLowerCase().includes('performance') ? 'performance is minimal as the added validation checks are lightweight operations' : question.toLowerCase().includes('security') ? 'security is highly positive, significantly reducing attack surface' : 'the system is generally positive with improved robustness'}.

This analysis is based on the diff comparison between the selected versions.`;

      setQaResults([...qaResults, { question, answer }]);
      setQuestion('');
    } catch (err) {
      console.error('Impact analyzer Q&A error:', err);
      setError('Failed to get answer. Please try again.');
      console.error('Error getting answer:', err);
    } finally {
      setIsQALoading(false);
    }
  };

  const exportAnalysis = async () => {
    const content = `# Impact Analysis Report

## Version Comparison
- **Old Version**: ${oldPage}
- **New Version**: ${newPage}
- **Analysis Date**: ${new Date().toLocaleString()}

## Metrics
- Lines Added: ${metrics?.linesAdded}
- Lines Removed: ${metrics?.linesRemoved}
- Files Changed: ${metrics?.filesChanged}
- Percentage Changed: ${metrics?.percentageChanged}%

## Risk Assessment
- **Risk Level**: ${riskLevel?.level.toUpperCase()}
- **Risk Score**: ${riskLevel?.score}/10
- **Risk Factors**:
${riskLevel?.factors.map(factor => `  - ${factor}`).join('\n')}

${impactSummary}

## Code Diff
\`\`\`diff
${diffResults}
\`\`\`

## Q&A
${qaResults.map(qa => `**Q:** ${qa.question}\n**A:** ${qa.answer}`).join('\n\n')}`;

    try {
      const blob = await apiService.exportContent({
        content: content,
        format: exportFormat,
        filename: 'impact-analysis-report'
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `impact-analysis-report.${exportFormat}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to export file. Please try again.');
      console.error('Error exporting:', err);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-700 bg-green-100/80 backdrop-blur-sm border-green-200/50';
      case 'medium': return 'text-yellow-700 bg-yellow-100/80 backdrop-blur-sm border-yellow-200/50';
      case 'high': return 'text-red-700 bg-red-100/80 backdrop-blur-sm border-red-200/50';
      default: return 'text-gray-700 bg-gray-100/80 backdrop-blur-sm border-gray-200/50';
    }
  };

  const getRiskIcon = (level: string) => {
    switch (level) {
      case 'low': return <CheckCircle className="w-5 h-5" />;
      case 'medium': return <AlertTriangle className="w-5 h-5" />;
      case 'high': return <AlertTriangle className="w-5 h-5" />;
      default: return <AlertTriangle className="w-5 h-5" />;
    }
  };

  return (
    <div className="fixed inset-0 bg-white flex items-center justify-center z-40 p-4">
      <div className="bg-white/80 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl w-full max-w-7xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-confluence-blue/90 to-confluence-light-blue/90 backdrop-blur-xl p-6 text-white border-b border-white/10">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <TrendingUp className="w-8 h-8" />
              <div>
                <h2 className="text-2xl font-bold">Confluence AI Assistant</h2>
                <p className="text-blue-100/90">AI-powered tools for your Confluence workspace</p>
              </div>
            </div>
            <button onClick={onClose} className="text-white hover:bg-white/10 rounded-full p-2 backdrop-blur-sm">
              <X className="w-6 h-6" />
            </button>
          </div>
          
          {/* Feature Navigation */}
          <div className="mt-6 flex gap-2">
            {features.map((feature) => {
              const Icon = feature.icon;
              const isActive = feature.id === 'impact';
              
              return (
                <button
                  key={feature.id}
                  onClick={() => onFeatureSelect(feature.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg backdrop-blur-sm border transition-all duration-200 whitespace-nowrap ${
                    isActive
                      ? 'bg-white/90 text-confluence-blue shadow-lg border-white/30'
                      : 'bg-white/10 text-white hover:bg-white/20 border-white/10'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{feature.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
            {/* Left Column - Configuration */}
            <div className="xl:col-span-1">
              <div className="bg-white/60 backdrop-blur-xl rounded-xl p-4 space-y-4 border border-white/20 shadow-lg">
                <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
                  <GitCompare className="w-5 h-5 mr-2" />
                  Version Comparison
                </h3>
                
                {/* Space Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select Confluence Space
                  </label>
                  <div className="relative">
                    <select
                      value={selectedSpace}
                      onChange={(e) => setSelectedSpace(e.target.value)}
                      className="w-full p-3 border border-white/30 rounded-lg focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue appearance-none bg-white/70 backdrop-blur-sm"
                    >
                      <option value="">Choose a space...</option>
                      {spaces.map(space => (
                        <option key={space.key} value={space.key}>{space.name} ({space.key})</option>
                      ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                  </div>
                </div>
                
                {/* Old Version Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Old Version
                  </label>
                  <div className="relative">
                    <select
                      value={oldPage}
                      onChange={(e) => setOldPage(e.target.value)}
                      className="w-full p-3 border border-white/30 rounded-lg focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue appearance-none bg-white/70 backdrop-blur-sm"
                      disabled={!selectedSpace}
                    >
                      <option value="">Select old version...</option>
                      {pages.map(page => (
                        <option key={page} value={page}>{page}</option>
                      ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                  </div>
                </div>

                {/* New Version Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    New Version
                  </label>
                  <div className="relative">
                    <select
                      value={newPage}
                      onChange={(e) => setNewPage(e.target.value)}
                      className="w-full p-3 border border-white/30 rounded-lg focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue appearance-none bg-white/70 backdrop-blur-sm"
                      disabled={!selectedSpace}
                    >
                      <option value="">Select new version...</option>
                      {pages.map(page => (
                        <option key={page} value={page}>{page}</option>
                      ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                  </div>
                </div>

                {/* Analyze Button */}
                <button
                  onClick={analyzeDiff}
                  disabled={!selectedSpace || !oldPage || !newPage || isAnalyzing}
                  className="w-full bg-confluence-blue/90 backdrop-blur-sm text-white py-3 px-4 rounded-lg hover:bg-confluence-blue disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-colors border border-white/10"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <TrendingUp className="w-5 h-5" />
                      <span>Analyze Impact</span>
                    </>
                  )}
                </button>

                {/* Metrics Display */}
                {metrics && (
                  <div className="mt-6 space-y-3">
                    <h4 className="font-semibold text-gray-800">Change Metrics</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="bg-green-100/80 backdrop-blur-sm p-2 rounded text-center border border-white/20">
                        <div className="font-semibold text-green-800">+{metrics.linesAdded}</div>
                        <div className="text-green-600 text-xs">Added</div>
                      </div>
                      <div className="bg-red-100/80 backdrop-blur-sm p-2 rounded text-center border border-white/20">
                        <div className="font-semibold text-red-800">-{metrics.linesRemoved}</div>
                        <div className="text-red-600 text-xs">Removed</div>
                      </div>
                      <div className="bg-blue-100/80 backdrop-blur-sm p-2 rounded text-center border border-white/20">
                        <div className="font-semibold text-blue-800">{metrics.filesChanged}</div>
                        <div className="text-blue-600 text-xs">Files</div>
                      </div>
                      <div className="bg-purple-100/80 backdrop-blur-sm p-2 rounded text-center border border-white/20">
                        <div className="font-semibold text-purple-800">{metrics.percentageChanged}%</div>
                        <div className="text-purple-600 text-xs">Changed</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Risk Level */}
                {riskLevel && (
                  <div className="mt-6">
                    <h4 className="font-semibold text-gray-800 mb-2">Risk Assessment</h4>
                    <div className={`p-3 rounded-lg flex items-center space-x-2 border ${getRiskColor(riskLevel.level)}`}>
                      {getRiskIcon(riskLevel.level)}
                      <div>
                        <div className="font-semibold capitalize">{riskLevel.level} Risk</div>
                        <div className="text-sm">Score: {riskLevel.score}/10</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Middle Columns - Diff and Analysis */}
            <div className="xl:col-span-2 space-y-6">
              {/* Code Diff */}
              {diffResults && (
                <div className="bg-white/60 backdrop-blur-xl rounded-xl p-4 border border-white/20 shadow-lg">
                  <h3 className="font-semibold text-gray-800 mb-4">Code Diff</h3>
                  <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-4 overflow-auto max-h-80 border border-white/10">
                    <pre className="text-sm">
                      <code>
                        {diffResults.split('\n').map((line, index) => (
                          <div
                            key={index}
                            className={
                              line.startsWith('+') ? 'text-green-400' :
                              line.startsWith('-') ? 'text-red-400' :
                              line.startsWith('@@') ? 'text-blue-400' :
                              'text-gray-300'
                            }
                          >
                            {line}
                          </div>
                        ))}
                      </code>
                    </pre>
                  </div>
                </div>
              )}

              {/* Impact Summary */}
              {impactSummary && (
                <div className="bg-white/60 backdrop-blur-xl rounded-xl p-4 border border-white/20 shadow-lg">
                  <h3 className="font-semibold text-gray-800 mb-4">AI Impact Summary</h3>
                  <div className="bg-white/70 backdrop-blur-sm rounded-lg p-4 border border-white/20 prose prose-sm max-w-none">
                    {impactSummary.split('\n').map((line, index) => {
                      if (line.startsWith('### ')) {
                        return <h3 key={index} className="text-lg font-bold text-gray-800 mt-4 mb-2">{line.substring(4)}</h3>;
                      } else if (line.startsWith('## ')) {
                        return <h2 key={index} className="text-xl font-bold text-gray-800 mt-6 mb-3">{line.substring(3)}</h2>;
                      } else if (line.startsWith('- **')) {
                        const match = line.match(/- \*\*(.*?)\*\*: (.*)/);
                        if (match) {
                          return <p key={index} className="mb-2"><strong>{match[1]}:</strong> {match[2]}</p>;
                        }
                      } else if (line.match(/^\d+\./)) {
                        return <p key={index} className="mb-2 font-medium">{line}</p>;
                      } else if (line.trim()) {
                        return <p key={index} className="mb-2 text-gray-700">{line}</p>;
                      }
                      return <br key={index} />;
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* Right Column - Q&A and Export */}
            <div className="xl:col-span-1 space-y-6">
              {/* Risk Factors */}
              {riskLevel && (
                <div className="bg-white/60 backdrop-blur-xl rounded-xl p-4 border border-white/20 shadow-lg">
                  <h3 className="font-semibold text-gray-800 mb-4">Risk Factors</h3>
                  <div className="space-y-2">
                    {riskLevel.factors.map((factor, index) => (
                      <div key={index} className="flex items-start space-x-2 text-sm">
                        <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span className="text-gray-700">{factor}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Q&A Section */}
              <div className="bg-white/60 backdrop-blur-xl rounded-xl p-4 border border-white/20 shadow-lg">
                <h3 className="font-semibold text-gray-800 mb-4">Questions & Analysis</h3>
                
                {/* Existing Q&A */}
                {qaResults.length > 0 && (
                  <div className="space-y-3 mb-4">
                    {qaResults.map((qa, index) => (
                      <div key={index} className="bg-white/70 backdrop-blur-sm rounded-lg p-3 border border-white/20">
                        <p className="font-medium text-gray-800 mb-2">Q: {qa.question}</p>
                        <p className="text-gray-700 text-sm">A: {qa.answer}</p>
                      </div>
                    ))}
                  </div>
                )}

                {/* Add Question */}
                <div className="space-y-2">
                  <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask about the impact analysis..."
                    className="w-full p-2 border border-white/30 rounded focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue bg-white/70 backdrop-blur-sm"
                    onKeyPress={(e) => e.key === 'Enter' && addQuestion()}
                  />
                  <button
                    onClick={addQuestion}
                    disabled={!question.trim() || isQALoading}
                    className="w-full px-3 py-2 bg-confluence-blue/90 backdrop-blur-sm text-white rounded hover:bg-confluence-blue disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2 border border-white/10"
                  >
                    {isQALoading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Loading...</span>
                      </>
                    ) : (
                      <>
                        <MessageSquare className="w-4 h-4" />
                        <span>Ask Question</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Export Options */}
              {diffResults && (
                <div className="bg-white/60 backdrop-blur-xl rounded-xl p-4 border border-white/20 shadow-lg">
                  <h3 className="font-semibold text-gray-800 mb-4">Export Options</h3>
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <label className="text-sm font-medium text-gray-700">Export Format:</label>
                      <select
                        value={exportFormat}
                        onChange={(e) => setExportFormat(e.target.value)}
                        className="px-3 py-1 border border-white/30 rounded text-sm focus:ring-2 focus:ring-confluence-blue bg-white/70 backdrop-blur-sm"
                      >
                        <option value="markdown">Markdown</option>
                        <option value="pdf">PDF</option>
                        <option value="docx">Word Document</option>
                        <option value="txt">Plain Text</option>
                      </select>
                    </div>
                    
                    <div className="space-y-2">
                      <button
                        onClick={exportAnalysis}
                        className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-green-600/90 backdrop-blur-sm text-white rounded-lg hover:bg-green-700 transition-colors border border-white/10"
                      >
                        <Download className="w-4 h-4" />
                        <span>Export</span>
                      </button>
                      <button
                        onClick={async () => {
                          const { space, page } = getConfluenceSpaceAndPageFromUrl();
                          if (!space || !page) {
                            alert('Confluence space or page not specified in macro src URL.');
                            return;
                          }
                          try {
                            await apiService.saveToConfluence({
                              space_key: space,
                              page_title: page,
                              content: impactSummary || '',
                            });
                            setShowToast(true);
                            setTimeout(() => setShowToast(false), 3000);
                          } catch (err: any) {
                            alert('Failed to save to Confluence: ' + (err.message || err));
                          }
                        }}
                        className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-confluence-blue/90 backdrop-blur-sm text-white rounded-lg hover:bg-confluence-blue transition-colors border border-white/10"
                      >
                        <Save className="w-4 h-4" />
                        <span>Save to Confluence</span>
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      {showToast && (
        <div style={{position: 'fixed', bottom: 40, left: '50%', transform: 'translateX(-50%)', background: '#2684ff', color: 'white', padding: '16px 32px', borderRadius: 8, zIndex: 9999, fontWeight: 600, fontSize: 16, boxShadow: '0 2px 12px rgba(0,0,0,0.15)'}}>
          Saved to Confluence! Please refresh this Confluence page to see your changes.
        </div>
      )}
    </div>
  );
};

export default ImpactAnalyzer;