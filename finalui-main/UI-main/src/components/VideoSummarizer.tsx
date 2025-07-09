import React, { useState, useEffect } from 'react';
import { Video, Download, Save, X, ChevronDown, ChevronRight, Loader2, Search, Code, TrendingUp, TestTube, MessageSquare, Check, ChevronUp, Image } from 'lucide-react';
import { FeatureType } from '../App';
import { apiService, Space } from '../services/api';
import { getConfluenceSpaceAndPageFromUrl } from '../utils/urlUtils';

interface VideoSummarizerProps {
  onClose: () => void;
  onFeatureSelect: (feature: FeatureType) => void;
  autoSpaceKey?: string | null;
  isSpaceAutoConnected?: boolean;
}

interface VideoContent {
  id: string;
  name: string;
  summary?: string;
  quotes?: string[];
  timestamps?: string[];
  qa?: { question: string; answer: string }[];
}

const VideoSummarizer: React.FC<VideoSummarizerProps> = ({ onClose, onFeatureSelect, autoSpaceKey, isSpaceAutoConnected }) => {
  const [selectedSpace, setSelectedSpace] = useState('');
  const [selectedPages, setSelectedPages] = useState<string[]>([]);
  const [videos, setVideos] = useState<VideoContent[]>([]);
  const [expandedVideo, setExpandedVideo] = useState<string | null>(null);
  const [newQuestion, setNewQuestion] = useState('');
  const [selectedVideo, setSelectedVideo] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isQALoading, setIsQALoading] = useState(false);
  const [exportFormat, setExportFormat] = useState('markdown');
  const [spaces, setSpaces] = useState<Space[]>([]);
  const [pages, setPages] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [isPageDropdownOpen, setIsPageDropdownOpen] = useState(false);
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
      setSelectedPages([]); // Reset selected pages when space changes
    } catch (err) {
      setError('Failed to load pages. Please check your space key.');
      console.error('Error loading pages:', err);
    }
  };

  const handlePageSelection = (page: string) => {
    setSelectedPages(prev => 
      prev.includes(page) 
        ? prev.filter(p => p !== page)
        : [...prev, page]
    );
  };

  const selectAllPages = () => {
    setSelectedPages(pages);
  };

  const clearAllPages = () => {
    setSelectedPages([]);
  };

  const processVideos = async () => {
    if (!selectedSpace || selectedPages.length === 0) {
      setError('Please select a space and at least one page.');
      return;
    }
    
    setIsProcessing(true);
    setError('');
    
    try {
      for (let i = 0; i < selectedPages.length; i++) {
        const page = selectedPages[i];
        
        try {
          const result = await apiService.videoSummarizer({
            space_key: selectedSpace,
            page_title: page
          });
          
          const newVideo: VideoContent = {
            id: Date.now().toString() + i, // Ensure unique IDs
            name: page,
            summary: result.summary,
            quotes: result.quotes,
            timestamps: result.timestamps,
            qa: result.qa
          };
          
          setVideos(prev => [...prev, newVideo]);
        } catch (err) {
          console.error(`Error processing page ${page}:`, err);
          // Continue with next page even if one fails
        }
      }
    } catch (err) {
      setError('Failed to process videos. Please try again.');
      console.error('Error processing videos:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const addQuestion = async () => {
    if (!newQuestion.trim() || !selectedVideo) {
      console.log('Missing question or selected video:', { newQuestion, selectedVideo });
      return;
    }
    
    console.log('Adding question:', newQuestion, 'for video:', selectedVideo);
    setIsQALoading(true);
    
    try {
      const result = await apiService.videoSummarizer({
        space_key: selectedSpace,
        page_title: selectedPages[0], // Use first selected page for Q&A
        question: newQuestion
      });

      console.log('Q&A API response:', result);

      const answer = result.answer || 'AI-generated answer based on the video content analysis...';
      
      setVideos(prev => prev.map(v => 
        v.id === selectedVideo 
          ? { 
              ...v, 
              qa: [...(v.qa || []), { question: newQuestion, answer: answer }]
            } 
          : v
      ));
      setNewQuestion('');
    } catch (err) {
      console.error('Q&A API error:', err);
      setError('Failed to get answer. Please try again.');
      console.error('Error getting answer:', err);
    } finally {
      setIsQALoading(false);
    }
  };

  const exportSummary = async (video: VideoContent, format: string) => {
    const content = `# Video Summary: ${video.name}

## Summary
${video.summary}

## Key Quotes
${video.quotes?.map(quote => `- "${quote}"`).join('\n')}

## Timestamps
${video.timestamps?.map(ts => `- ${ts}`).join('\n')}

## Q&A
${video.qa?.map(qa => `**Q:** ${qa.question}\n**A:** ${qa.answer}`).join('\n\n')}`;

    try {
      const blob = await apiService.exportContent({
        content: content,
        format: format,
        filename: `${video.name.replace(/\s+/g, '_')}_summary`
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${video.name.replace(/\s+/g, '_')}_summary.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to export file. Please try again.');
      console.error('Error exporting:', err);
    }
  };

  const exportAllVideos = async () => {
    if (videos.length === 0) return;

    const allContent = videos.map(video => {
      return `# Video Summary: ${video.name}

## Summary
${video.summary}

## Key Quotes
${video.quotes?.map(quote => `- "${quote}"`).join('\n')}

## Timestamps
${video.timestamps?.map(ts => `- ${ts}`).join('\n')}

## Q&A
${video.qa?.map(qa => `**Q:** ${qa.question}\n**A:** ${qa.answer}`).join('\n\n')}

---`;
    }).join('\n\n');

    try {
      const blob = await apiService.exportContent({
        content: allContent,
        format: exportFormat,
        filename: `all_video_summaries`
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `all_video_summaries.${exportFormat}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to export all videos. Please try again.');
      console.error('Error exporting all videos:', err);
    }
  };

  return (
    <div className="fixed inset-0 bg-white flex items-center justify-center z-40 p-4">
      <div className="bg-white/80 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl w-full max-w-6xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-confluence-blue/90 to-confluence-light-blue/90 backdrop-blur-xl p-6 text-white border-b border-white/10">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Video className="w-8 h-8" />
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
              const isActive = feature.id === 'video';
              
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
          {error && (
            <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
              {error}
            </div>
          )}

          {/* Video Selection Section */}
          <div className="mb-6 bg-white/60 backdrop-blur-xl rounded-xl p-6 border border-white/20 shadow-lg">
            <h3 className="font-semibold text-gray-800 mb-4">Select Video Content</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                  <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                </div>
              </div>

              {/* Page Selection - Aesthetic Multiselect */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Video Pages ({selectedPages.length} selected)
                </label>
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setIsPageDropdownOpen(!isPageDropdownOpen)}
                    disabled={!selectedSpace}
                    className="w-full p-3 border border-white/30 rounded-lg focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue bg-white/70 backdrop-blur-sm text-left flex items-center justify-between disabled:bg-gray-100 disabled:cursor-not-allowed"
                  >
                    <span className={selectedPages.length === 0 ? 'text-gray-500' : 'text-gray-700'}>
                      {selectedPages.length === 0 
                        ? 'Choose pages...' 
                        : selectedPages.length === 1 
                          ? selectedPages[0]
                          : `${selectedPages.length} pages selected`
                      }
                    </span>
                    {isPageDropdownOpen ? (
                      <ChevronUp className="w-5 h-5 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-gray-400" />
                    )}
                  </button>

                  {/* Dropdown */}
                  {isPageDropdownOpen && selectedSpace && (
                    <div className="absolute z-10 w-full mt-1 bg-white/95 backdrop-blur-xl border border-white/30 rounded-lg shadow-xl max-h-60 overflow-hidden">
                      {/* Header with Select All/Clear All */}
                      <div className="p-3 border-b border-white/20 bg-white/50">
                        <div className="flex justify-between items-center">
                          <button
                            onClick={selectAllPages}
                            className="text-sm text-confluence-blue hover:text-confluence-blue/80 font-medium"
                          >
                            Select All
                          </button>
                          <button
                            onClick={clearAllPages}
                            className="text-sm text-gray-500 hover:text-gray-700 font-medium"
                          >
                            Clear All
                          </button>
                        </div>
                      </div>

                      {/* Page List */}
                      <div className="max-h-48 overflow-y-auto">
                        {pages.length === 0 ? (
                          <div className="p-3 text-gray-500 text-sm text-center">
                            No pages found in this space
                          </div>
                        ) : (
                          pages.map(page => (
                            <label
                              key={page}
                              className="flex items-center space-x-3 p-3 hover:bg-white/50 cursor-pointer border-b border-white/10 last:border-b-0"
                            >
                              <div className="relative">
                                <input
                                  type="checkbox"
                                  checked={selectedPages.includes(page)}
                                  onChange={() => handlePageSelection(page)}
                                  className="sr-only"
                                />
                                <div className={`w-5 h-5 border-2 rounded flex items-center justify-center transition-colors ${
                                  selectedPages.includes(page)
                                    ? 'bg-confluence-blue border-confluence-blue'
                                    : 'border-gray-300 hover:border-confluence-blue/50'
                                }`}>
                                  {selectedPages.includes(page) && (
                                    <Check className="w-3 h-3 text-white" />
                                  )}
                                </div>
                              </div>
                              <span className="text-sm text-gray-700 flex-1">{page}</span>
                            </label>
                          ))
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <button
              onClick={processVideos}
              disabled={!selectedSpace || selectedPages.length === 0 || isProcessing}
              className="mt-4 w-full bg-confluence-blue/90 backdrop-blur-sm text-white py-3 px-4 rounded-lg hover:bg-confluence-blue disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-colors border border-white/10"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Processing Videos...</span>
                </>
              ) : (
                <>
                  <Video className="w-5 h-5" />
                  <span>Process {selectedPages.length} Video{selectedPages.length !== 1 ? 's' : ''}</span>
                </>
              )}
            </button>
          </div>

          {/* Videos List */}
          <div className="space-y-4">
            {videos.map(video => (
              <div key={video.id} className="border border-white/30 rounded-xl overflow-hidden bg-white/60 backdrop-blur-xl shadow-lg">
                <div 
                  className="p-4 bg-white/50 backdrop-blur-sm cursor-pointer hover:bg-white/70 transition-colors"
                  onClick={() => setExpandedVideo(expandedVideo === video.id ? null : video.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="w-12 h-12 bg-confluence-light-blue/20 backdrop-blur-sm rounded-lg flex items-center justify-center border border-white/20">
                        <Video className="w-6 h-6 text-confluence-blue" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-800">{video.name}</h4>
                        <div className="flex items-center space-x-4 text-sm text-gray-500">
                          <span>Processed</span>
                          <span className="px-2 py-1 rounded-full text-xs bg-green-100/80 backdrop-blur-sm text-green-800 border border-white/20">
                            Completed
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <button
                          onClick={(e) => { e.stopPropagation(); exportSummary(video, exportFormat); }}
                          className="px-3 py-1 bg-confluence-blue/90 backdrop-blur-sm text-white rounded text-sm hover:bg-confluence-blue transition-colors border border-white/10"
                        >
                          Export
                        </button>
                      </div>
                      {expandedVideo === video.id ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
                    </div>
                  </div>
                </div>

                {expandedVideo === video.id && (
                  <div className="border-t border-white/20 bg-white/40 backdrop-blur-xl">
                    <div className="p-6 space-y-6">
                      {/* Summary */}
                      <div>
                        <h5 className="font-semibold text-gray-800 mb-3">AI Summary</h5>
                        <div className="bg-white/70 backdrop-blur-sm rounded-lg p-4 border border-white/20">
                          <p className="text-gray-700">{video.summary}</p>
                        </div>
                      </div>

                      {/* Key Quotes */}
                      {video.quotes && video.quotes.length > 0 && (
                        <div>
                          <h5 className="font-semibold text-gray-800 mb-3">Key Quotes</h5>
                          <div className="space-y-2">
                            {video.quotes.map((quote, index) => (
                              <div key={index} className="bg-white/70 backdrop-blur-sm rounded-lg p-4 border-l-4 border-confluence-blue border border-white/20">
                                <p className="text-gray-700 italic">"{quote}"</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Timestamps Section */}
                      {video.timestamps && video.timestamps.length > 0 && (
                        <div>
                          <h5 className="font-semibold text-gray-800 mb-3">Timestamps</h5>
                          <div className="space-y-2">
                            {video.timestamps.map((ts, index) => (
                              <div
                                key={index}
                                className="bg-white/70 backdrop-blur-sm rounded-lg p-4 border-l-4 border-yellow-500 border border-white/20"
                              >
                                <p className="text-gray-700">{ts}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Q&A Section */}
                      <div>
                        <h5 className="font-semibold text-gray-800 mb-3">Questions & Answers</h5>
                        <div className="space-y-4">
                          {video.qa && video.qa.length > 0 && (
                            <div className="space-y-3">
                              {video.qa.map((qa, index) => (
                                <div key={index} className="bg-white/70 backdrop-blur-sm rounded-lg p-4 border border-white/20">
                                  <p className="font-medium text-gray-800 mb-2">Q: {qa.question}</p>
                                  <p className="text-gray-700">A: {qa.answer}</p>
                                </div>
                              ))}
                            </div>
                          )}
                          
                          {/* Add New Question */}
                          <div className="bg-white/70 backdrop-blur-sm rounded-lg p-4 border border-white/20">
                            <div className="flex space-x-2">
                              <input
                                type="text"
                                value={newQuestion}
                                onChange={(e) => setNewQuestion(e.target.value)}
                                placeholder="Ask a question about this video..."
                                className="flex-1 p-2 border border-white/30 rounded focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue bg-white/70 backdrop-blur-sm"
                                onKeyPress={(e) => e.key === 'Enter' && addQuestion()}
                              />
                              <button
                                onClick={() => {
                                  setSelectedVideo(video.id);
                                  addQuestion();
                                }}
                                disabled={isQALoading}
                                className="px-4 py-2 bg-confluence-blue/90 backdrop-blur-sm text-white rounded hover:bg-confluence-blue disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center border border-white/10"
                              >
                                {isQALoading ? (
                                  <>
                                    <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                                    <span>Loading...</span>
                                  </>
                                ) : (
                                  <>
                                    <MessageSquare className="w-4 h-4 mr-1" />
                                    <span>Ask</span>
                                  </>
                                )}
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Export Options */}
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
                        
                        <div className="flex space-x-2 pt-4 border-t border-white/20">
                          <button
                            onClick={() => exportSummary(video, exportFormat)}
                            className="flex items-center space-x-2 px-4 py-2 bg-green-600/90 backdrop-blur-sm text-white rounded-lg hover:bg-green-700 transition-colors border border-white/10"
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
                                  content: video.summary || '',
                                });
                                setShowToast(true);
                                setTimeout(() => setShowToast(false), 3000);
                              } catch (err: any) {
                                alert('Failed to save to Confluence: ' + (err.message || err));
                              }
                            }}
                            className="flex items-center space-x-2 px-4 py-2 bg-confluence-blue/90 backdrop-blur-sm text-white rounded-lg hover:bg-confluence-blue transition-colors border border-white/10"
                          >
                            <Save className="w-4 h-4" />
                            <span>Save to Confluence</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Bulk Export Section - Show when 2 or more videos */}
          {videos.length >= 2 && (
            <div className="mt-6 bg-white/60 backdrop-blur-xl rounded-xl p-6 border border-white/20 shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-gray-800 mb-2">Export All Videos</h3>
                  <p className="text-sm text-gray-600">Export all {videos.length} processed videos in a single file</p>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <label className="text-sm font-medium text-gray-700">Format:</label>
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
                  <button
                    onClick={exportAllVideos}
                    className="flex items-center space-x-2 px-4 py-2 bg-green-600/90 backdrop-blur-sm text-white rounded-lg hover:bg-green-700 transition-colors border border-white/10"
                  >
                    <Download className="w-4 h-4" />
                    <span>Export All ({videos.length})</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {videos.length === 0 && !isProcessing && (
            <div className="text-center py-12">
              <Video className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-600 mb-2">No Videos Processed</h3>
              <p className="text-gray-500">Select a space and pages with video content to start generating AI summaries.</p>
            </div>
          )}
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

export default VideoSummarizer;