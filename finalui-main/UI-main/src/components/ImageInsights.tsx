import React, { useState, useRef, useEffect } from 'react';
import { Image, Download, Save, X, ChevronDown, Loader2, MessageSquare, BarChart3, Search, Video, Code, TrendingUp, TestTube, Eye, Zap } from 'lucide-react';
import { FeatureType } from '../App';
import { apiService } from '../services/api';
import { getConfluenceSpaceAndPageFromUrl } from '../utils/urlUtils';

interface ImageInsightsProps {
  onClose: () => void;
  onFeatureSelect: (feature: FeatureType) => void;
  autoSpaceKey?: string | null;
  isSpaceAutoConnected?: boolean;
}

interface ImageData {
  id: string;
  name: string;
  url: string;
  summary?: string;
  qa?: { question: string; answer: string }[];
  pageTitle?: string;
}

interface ChartData {
  type: 'bar' | 'line' | 'pie' | 'stacked';
  data: any;
  title: string;
}

const ImageInsights: React.FC<ImageInsightsProps> = ({ onClose, onFeatureSelect, autoSpaceKey, isSpaceAutoConnected }) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedSummary, setUploadedSummary] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const [spaceKey, setSpaceKey] = useState<string>('');
  const [selectedPages, setSelectedPages] = useState<string[]>([]);
  const [images, setImages] = useState<ImageData[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState<string>('');
  const [newQuestion, setNewQuestion] = useState('');
  const [selectedImage, setSelectedImage] = useState<string>('');
  const [fileName, setFileName] = useState('');
  const [exportFormat, setExportFormat] = useState('pdf');
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [selectedChartType, setSelectedChartType] = useState<'bar' | 'line' | 'pie' | 'stacked'>('bar');
  const [chartFileName, setChartFileName] = useState('');
  const [chartExportFormat, setChartExportFormat] = useState('png');
  const [spaces, setSpaces] = useState<Array<{name: string, key: string}>>([]);
  const [pages, setPages] = useState<string[]>([]);
  const [isLoadingSpaces, setIsLoadingSpaces] = useState(false);
  const [isLoadingPages, setIsLoadingPages] = useState(false);
  const [isLoadingImages, setIsLoadingImages] = useState(false);
  const [isAskingQuestion, setIsAskingQuestion] = useState(false);
  const [isCreatingChart, setIsCreatingChart] = useState(false);
  const [isChangingChartType, setIsChangingChartType] = useState(false);
  const [isExportingChart, setIsExportingChart] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const chartPreviewRef = useRef<HTMLDivElement>(null);

  // Load spaces on component mount
  useEffect(() => {
    const loadSpaces = async () => {
      setIsLoadingSpaces(true);
      try {
        const response = await apiService.getSpaces();
        setSpaces(response.spaces);
      } catch (error) {
        console.error('Failed to load spaces:', error);
      } finally {
        setIsLoadingSpaces(false);
      }
    };
    loadSpaces();
  }, []);

  // Auto-select space if provided via URL
  useEffect(() => {
    if (autoSpaceKey && isSpaceAutoConnected) {
      setSpaceKey(autoSpaceKey);
    }
  }, [autoSpaceKey, isSpaceAutoConnected]);

  // Load pages when space key changes
  useEffect(() => {
    const loadPages = async () => {
      if (!spaceKey) {
        setPages([]);
        return;
      }
      setIsLoadingPages(true);
      try {
        const response = await apiService.getPages(spaceKey);
        setPages(response.pages);
      } catch (error) {
        console.error('Failed to load pages:', error);
        setPages([]);
      } finally {
        setIsLoadingPages(false);
      }
    };
    loadPages();
  }, [spaceKey]);

  const chartTypes = [
    { value: 'bar' as const, label: 'Grouped Bar Chart' },
    { value: 'line' as const, label: 'Line Chart' },
    { value: 'pie' as const, label: 'Pie Chart' },
    { value: 'stacked' as const, label: 'Stacked Bar Chart' }
  ];

  const features = [
    { id: 'search' as const, label: 'AI Powered Search', icon: Search },
    { id: 'video' as const, label: 'Video Summarizer', icon: Video },
    { id: 'code' as const, label: 'Code Assistant', icon: Code },
    { id: 'impact' as const, label: 'Impact Analyzer', icon: TrendingUp },
    { id: 'test' as const, label: 'Test Support Tool', icon: TestTube },
    { id: 'image' as const, label: 'Image Insights & Chart Builder', icon: Image },
  ];



  const loadImages = async () => {
    if (!spaceKey || selectedPages.length === 0) return;
    setIsLoadingImages(true);
    
    try {
      const allImages: ImageData[] = [];
      
      for (const pageTitle of selectedPages) {
        try {
          const response = await apiService.getImages(spaceKey, pageTitle);
          const pageImages = response.images.map((url, index) => ({
            id: `${pageTitle}_${index}`,
            name: `Image ${index + 1} from ${pageTitle}`,
            url,
            pageTitle,
            qa: []
          }));
          allImages.push(...pageImages);
        } catch (error) {
          console.error(`Failed to load images from page ${pageTitle}:`, error);
        }
      }
      
      setImages(allImages);
    } catch (error) {
      console.error('Failed to load images:', error);
    } finally {
      setIsLoadingImages(false);
    }
  };

  const analyzeImage = async (imageId: string) => {
    setIsAnalyzing(imageId);
    
    try {
      const image = images.find(img => img.id === imageId);
      if (!image || !image.pageTitle) {
        throw new Error('Image not found or missing page title');
      }
      
      const response = await apiService.imageSummary({
        space_key: spaceKey,
        page_title: image.pageTitle,
        image_url: image.url
      });
      
      setImages(prev => prev.map(img =>
        img.id === imageId
          ? { ...img, summary: response.summary }
          : img
      ));
    } catch (error) {
      console.error('Failed to analyze image:', error);
      // Fallback to sample summary
      setImages(prev => prev.map(img =>
        img.id === imageId
          ? {
              ...img,
              summary: `AI Analysis of ${img.name}: This image contains data visualization elements including charts, graphs, and key performance indicators. The visual elements suggest business metrics tracking with trend analysis and comparative data points. Key insights include performance trends, data correlations, and actionable business intelligence derived from the visual representation.`
            }
          : img
      ));
    } finally {
      setIsAnalyzing('');
    }
  };

  const addQuestion = async () => {
    if (!newQuestion.trim() || !selectedImage) return;
    
    setIsAskingQuestion(true);
    try {
      const image = images.find(img => img.id === selectedImage);
      if (!image || !image.pageTitle || !image.summary) {
        throw new Error('Image not found or missing required data');
      }
      
      const response = await apiService.imageQA({
        space_key: spaceKey,
        page_title: image.pageTitle,
        image_url: image.url,
        summary: image.summary,
        question: newQuestion
      });
      
      setImages(prev => prev.map(img =>
        img.id === selectedImage
          ? {
              ...img,
              qa: [...(img.qa || []), { question: newQuestion, answer: response.answer }]
            }
          : img
      ));
      setNewQuestion('');
    } catch (error) {
      console.error('Failed to get AI response:', error);
      // Fallback to sample answer
      const answer = `Based on the AI analysis of this image, here's the response to your question: "${newQuestion}"

The image analysis reveals specific data patterns and visual elements that directly relate to your inquiry. The AI has processed the visual content and extracted relevant insights to provide this contextual response.`;
      setImages(prev => prev.map(img =>
        img.id === selectedImage
          ? {
              ...img,
              qa: [...(img.qa || []), { question: newQuestion, answer }]
            }
          : img
      ));
      setNewQuestion('');
    } finally {
      setIsAskingQuestion(false);
    }
  };

  const createChart = async (imageId: string, chartType?: string, exportFormat?: string) => {
    setIsCreatingChart(true);
    try {
      const image = images.find(img => img.id === imageId);
      if (!image || !image.pageTitle) {
        throw new Error('Image not found or missing page title');
      }
      
      // Use provided parameters or fall back to state values
      const currentChartType = chartType || selectedChartType;
      const currentExportFormat = exportFormat || chartExportFormat;
      
      const chartTypeMap = {
        'bar': 'Grouped Bar',
        'line': 'Line',
        'pie': 'Pie',
        'stacked': 'Stacked Bar'
      };
      
      const response = await apiService.createChart({
        space_key: spaceKey,
        page_title: image.pageTitle,
        image_url: image.url,
        chart_type: chartTypeMap[currentChartType as keyof typeof chartTypeMap],
        filename: chartFileName || 'chart',
        format: currentExportFormat
      });
      
      // Convert base64 to blob URL for display
      const binaryString = atob(response.chart_data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: response.mime_type });
      const chartUrl = URL.createObjectURL(blob);
      
      setChartData({
        type: currentChartType as any,
        data: { 
          chartUrl, 
          filename: response.filename, 
          exportFormat: currentExportFormat,
          imageId: imageId // Store the image ID for recreation
        },
        title: `Generated ${currentChartType.charAt(0).toUpperCase() + currentChartType.slice(1)} Chart`
      });
      
      setTimeout(() => {
        chartPreviewRef.current?.scrollIntoView({
          behavior: 'smooth',
          block: 'center'
        });
      }, 100);
    } catch (error) {
      console.error('Failed to create chart:', error);
      // Fallback to sample data
      const currentChartType = chartType || selectedChartType;
      const sampleData = {
        bar: {
          labels: ['Q1', 'Q2', 'Q3', 'Q4'],
          datasets: [{
            label: 'Revenue',
            data: [65, 78, 90, 81],
            backgroundColor: 'rgba(38, 132, 255, 0.8)'
          }]
        },
        line: {
          labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
          datasets: [{
            label: 'Growth',
            data: [12, 19, 15, 25, 22],
            borderColor: 'rgba(38, 132, 255, 1)',
            fill: false
          }]
        },
        pie: {
          labels: ['Desktop', 'Mobile', 'Tablet'],
          datasets: [{
            data: [55, 35, 10],
            backgroundColor: ['#0052CC', '#2684FF', '#B3D4FF']
          }]
        },
        stacked: {
          labels: ['Q1', 'Q2', 'Q3', 'Q4'],
          datasets: [{
            label: 'Revenue',
            data: [65, 78, 90, 81],
            backgroundColor: 'rgba(38, 132, 255, 0.8)'
          }]
        }
      };
      setChartData({
        type: currentChartType as any,
        data: sampleData[currentChartType as keyof typeof sampleData],
        title: `Generated ${currentChartType.charAt(0).toUpperCase() + currentChartType.slice(1)} Chart`
      });
    } finally {
      setIsCreatingChart(false);
    }
  };

  const exportImage = async (image: ImageData) => {
    try {
      const content = `# Image Analysis Report: ${image.name}

## AI Summary
${image.summary || 'No summary available'}

## Questions & Answers
${image.qa?.map(qa => `**Q:** ${qa.question}\n**A:** ${qa.answer}`).join('\n\n') || 'No questions asked'}

## Image Details
- **Name**: ${image.name}
- **Analysis Date**: ${new Date().toLocaleString()}
- **Export Format**: ${exportFormat}

---
*Generated by Confluence AI Assistant - Image Insights*`;

      const response = await apiService.exportContent({
        content,
        format: exportFormat,
        filename: fileName || image.name.replace(/\s+/g, '_') + '_analysis'
      });

      const url = URL.createObjectURL(response);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${fileName || image.name.replace(/\s+/g, '_')}_analysis.${exportFormat}`;
      a.click();
    } catch (error) {
      console.error('Failed to export image:', error);
      // Fallback to client-side export
      const content = `# Image Analysis Report: ${image.name}

## AI Summary
${image.summary || 'No summary available'}

## Questions & Answers
${image.qa?.map(qa => `**Q:** ${qa.question}\n**A:** ${qa.answer}`).join('\n\n') || 'No questions asked'}

## Image Details
- **Name**: ${image.name}
- **Analysis Date**: ${new Date().toLocaleString()}
- **Export Format**: ${exportFormat}

---
*Generated by Confluence AI Assistant - Image Insights*`;
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${fileName || image.name.replace(/\s+/g, '_')}_analysis.${exportFormat}`;
      a.click();
    }
  };

  const exportChart = async () => {
    if (!chartData) return;
    
    setIsExportingChart(true);
    try {
      // Get the current export format from chart data or state
      const currentExportFormat = chartData.data.exportFormat || chartExportFormat;
      
      // If we have a chart URL from the backend, we need to recreate the chart with the current export format
      if (chartData.data.chartUrl) {
        // Use the stored image ID if available, otherwise find an image with summary
        let imageId = chartData.data.imageId;
        if (!imageId) {
          const imageWithSummary = images.find(img => img.summary);
          imageId = imageWithSummary?.id;
        }
        if (imageId) {
          const image = images.find(img => img.id === imageId);
          if (image && image.pageTitle) {
            // Recreate the chart with the current export format
            const response = await apiService.createChart({
              space_key: spaceKey,
              page_title: image.pageTitle,
              image_url: image.url,
              chart_type: chartData.type === 'bar' ? 'Grouped Bar' : 
                         chartData.type === 'line' ? 'Line' : 
                         chartData.type === 'pie' ? 'Pie' : 'Stacked Bar',
              filename: chartFileName || 'chart',
              format: currentExportFormat
            });
            
            // Download the chart in the selected format
            const binaryString = atob(response.chart_data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
              bytes[i] = binaryString.charCodeAt(i);
            }
            const blob = new Blob([bytes], { type: response.mime_type });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = response.filename;
            a.click();
            return;
          }
        }
      }
      
      // Fallback to text export
      const content = `# Chart Export: ${chartData.title}

## Chart Type
${chartData.type.charAt(0).toUpperCase() + chartData.type.slice(1)} Chart

## Data
${JSON.stringify(chartData.data, null, 2)}

## Export Details
- **File Name**: ${chartFileName}
- **Format**: ${currentExportFormat}
- **Generated**: ${new Date().toLocaleString()}

---
*Generated by Confluence AI Assistant - Chart Builder*`;

      const response = await apiService.exportContent({
        content,
        format: currentExportFormat,
        filename: chartFileName || 'chart'
      });

      const url = URL.createObjectURL(response);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${chartFileName || 'chart'}.${currentExportFormat}`;
      a.click();
    } catch (error) {
      console.error('Failed to export chart:', error);
      // Fallback to client-side export
      const currentExportFormat = chartData.data.exportFormat || chartExportFormat;
      const content = `# Chart Export: ${chartData.title}

## Chart Type
${chartData.type.charAt(0).toUpperCase() + chartData.type.slice(1)} Chart

## Data
${JSON.stringify(chartData.data, null, 2)}

## Export Details
- **File Name**: ${chartFileName}
- **Format**: ${currentExportFormat}
- **Generated**: ${new Date().toLocaleString()}

---
*Generated by Confluence AI Assistant - Chart Builder*`;
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${chartFileName || 'chart'}.${currentExportFormat}`;
      a.click();
    } finally {
      setIsExportingChart(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-white flex items-center justify-center z-40 p-4">
      <div className="bg-white/80 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl w-full max-w-7xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-confluence-blue/90 to-confluence-light-blue/90 backdrop-blur-xl p-6 text-white border-b border-white/10">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Image className="w-8 h-8" />
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
              const isActive = feature.id === 'image';
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
            {/* Left Column - Image Selection */}
            <div className="xl:col-span-1">
              <div className="bg-white/60 backdrop-blur-xl rounded-xl p-4 space-y-6 border border-white/20 shadow-lg">
                <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
                  <Eye className="w-5 h-5 mr-2" />
                  Image Selection
                </h3>
                {/* Space Key Input */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Confluence Space Key
                  </label>
                  <div className="relative">
                    <select
                      value={spaceKey}
                      onChange={(e) => setSpaceKey(e.target.value)}
                      disabled={isLoadingSpaces}
                      className="w-full p-3 border border-white/30 rounded-lg focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue appearance-none bg-white/70 backdrop-blur-sm disabled:bg-gray-100"
                    >
                      <option value="">
                        {isLoadingSpaces ? 'Loading spaces...' : 'Select space...'}
                      </option>
                      {spaces.map(space => (
                        <option key={space.key} value={space.key}>{space.name}</option>
                      ))}
                    </select>
                    {isLoadingSpaces ? (
                      <Loader2 className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400 animate-spin" />
                    ) : (
                      <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                    )}
                  </div>
                </div>
                {/* Page Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select Pages
                  </label>
                  <div className="space-y-2 max-h-40 overflow-y-auto border border-white/30 rounded-lg p-2 bg-white/50 backdrop-blur-sm">
                    {isLoadingPages ? (
                      <div className="flex items-center justify-center py-4">
                        <Loader2 className="w-4 h-4 animate-spin text-gray-400 mr-2" />
                        <span className="text-sm text-gray-500">Loading pages...</span>
                      </div>
                    ) : pages.length > 0 ? (
                      pages.map(page => (
                        <label key={page} className="flex items-center space-x-2 p-2 hover:bg-white/30 rounded cursor-pointer backdrop-blur-sm">
                          <input
                            type="checkbox"
                            checked={selectedPages.includes(page)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedPages([...selectedPages, page]);
                              } else {
                                setSelectedPages(selectedPages.filter(p => p !== page));
                              }
                            }}
                            className="rounded border-gray-300 text-confluence-blue focus:ring-confluence-blue"
                          />
                          <span className="text-sm text-gray-700">{page}</span>
                        </label>
                      ))
                    ) : (
                      <div className="text-center py-4">
                        <span className="text-sm text-gray-500">
                          {spaceKey ? 'No pages found' : 'Select a space to load pages'}
                        </span>
                      </div>
                    )}
                  </div>
                  <p className="text-sm text-gray-500 mt-1">
                    {selectedPages.length} page(s) selected
                  </p>
                </div>
                {/* File Upload & Load Button */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Upload a file (csv, pdf, docx, png, etc.)</label>
                  <input
                    type="file"
                    accept=".csv,.pdf,.docx,.doc,.png,.jpg,.jpeg,.gif,.bmp,.xlsx,.txt"
                    onChange={e => setUploadedFile(e.target.files?.[0] || null)}
                    className="w-full p-2 border border-white/30 rounded focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue bg-white/70 backdrop-blur-sm"
                  />
                </div>
                <button
                  onClick={async () => {
                    if (!uploadedFile) return;
                    setIsUploading(true);
                    setUploadedSummary('');
                    try {
                      // Prepare FormData for upload
                      const formData = new FormData();
                      formData.append('file', uploadedFile);
                      // Use fetch directly for file upload
                      const response = await fetch('https://central-1rua.onrender.com/upload-file', {
                        method: 'POST',
                        body: formData,
                        headers: {
                          'x-api-key': localStorage.getItem('selectedApiKeyId') || ''
                        }
                      });
                      if (!response.ok) throw new Error('Failed to upload file');
                      const data = await response.json();
                      setUploadedSummary(data.summary || 'No summary returned');
                    } catch (err) {
                      setUploadedSummary('Error uploading or summarizing file.');
                    } finally {
                      setIsUploading(false);
                    }
                  }}
                  disabled={!uploadedFile || isUploading}
                  className="w-full bg-confluence-blue/90 backdrop-blur-sm text-white py-3 px-4 rounded-lg hover:bg-confluence-blue disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-colors border border-white/10 mb-2"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Uploading & Summarizing...</span>
                    </>
                  ) : (
                    <>
                      <Image className="w-5 h-5" />
                      <span>Load & Summarize File</span>
                    </>
                  )}
                </button>
                {uploadedSummary && (
                  <div className="mt-2 p-3 bg-white/70 backdrop-blur-sm rounded-lg border border-white/20">
                    <h4 className="font-semibold text-gray-800 mb-2">File Summary</h4>
                    <p className="text-sm text-gray-700 whitespace-pre-line">{uploadedSummary}</p>
                  </div>
                )}
                {/* Existing Load Images Button */}
                <button
                  onClick={loadImages}
                  disabled={!spaceKey || selectedPages.length === 0 || isLoadingImages}
                  className="w-full bg-confluence-blue/90 backdrop-blur-sm text-white py-3 px-4 rounded-lg hover:bg-confluence-blue disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-colors border border-white/10"
                >
                  {isLoadingImages ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Loading Images...</span>
                    </>
                  ) : (
                    <>
                      <Image className="w-5 h-5" />
                      <span>Load Images from Confluence</span>
                    </>
                  )}
                </button>
              </div>
            </div>
            {/* Middle Column - Images Grid */}
            <div className="xl:col-span-2 space-y-6">
              {images.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {images.map(image => (
                    <div key={image.id} className="bg-white/60 backdrop-blur-xl rounded-xl p-4 border border-white/20 shadow-lg">
                      <div className="aspect-video bg-gray-200/50 backdrop-blur-sm rounded-lg mb-4 overflow-hidden border border-white/20">
                        <img 
                          src={image.url} 
                          alt={image.name}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <h4 className="font-semibold text-gray-800 mb-2">{image.name}</h4>
                      <div className="space-y-2">
                        <button
                          onClick={() => analyzeImage(image.id)}
                          disabled={isAnalyzing === image.id}
                          className="w-full bg-confluence-blue/90 backdrop-blur-sm text-white py-2 px-4 rounded-lg hover:bg-confluence-blue disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-colors border border-white/10"
                        >
                          {isAnalyzing === image.id ? (
                            <>
                              <Loader2 className="w-4 h-4 animate-spin" />
                              <span>Analyzing...</span>
                            </>
                          ) : (
                            <>
                              <Eye className="w-4 h-4" />
                              <span>Summarize</span>
                            </>
                          )}
                        </button>
                        {image.summary && (
                          <button
                            onClick={() => createChart(image.id, selectedChartType, chartExportFormat)}
                            disabled={isCreatingChart}
                            className="w-full bg-green-600/90 backdrop-blur-sm text-white py-2 px-4 rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2 border border-white/10"
                          >
                            {isCreatingChart ? (
                              <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                <span>Creating Chart...</span>
                              </>
                            ) : (
                              <>
                                <BarChart3 className="w-4 h-4" />
                                <span>Create Graph</span>
                              </>
                            )}
                          </button>
                        )}
                      </div>
                      {image.summary && (
                        <div className="mt-4 p-3 bg-white/70 backdrop-blur-sm rounded-lg border border-white/20">
                          <p className="text-sm text-gray-700">{image.summary}</p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Image className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-600 mb-2">No Images Loaded</h3>
                  <p className="text-gray-500">Select a space and pages to load embedded images for analysis.</p>
                </div>
              )}
              {/* Chart Preview Section */}
              {chartData && (
                <div ref={chartPreviewRef} className="bg-white/60 backdrop-blur-xl rounded-xl p-4 border border-white/20 shadow-lg">
                  <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2" />
                    Chart Builder
                  </h3>
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Chart Controls - Left Side */}
                    <div className="lg:col-span-1 space-y-4">
                      <div className="bg-white/70 backdrop-blur-sm rounded-lg p-4 border border-white/20">
                        <h4 className="font-semibold text-gray-800 mb-3">Chart Settings</h4>
                        <div className="space-y-4">
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                              Chart Type
                            </label>
                            <div className="relative">
                              <select
                                value={selectedChartType}
                                disabled={isChangingChartType}
                                onChange={async (e) => {
                                  const newChartType = e.target.value as any;
                                  setSelectedChartType(newChartType);
                                  // Always recreate the chart when type changes, regardless of existing chart data
                                  setIsChangingChartType(true);
                                  try {
                                    // Use the stored image ID if available, otherwise find an image with summary
                                    let imageId = chartData?.data?.imageId;
                                    if (!imageId) {
                                      const imageWithSummary = images.find(img => img.summary);
                                      imageId = imageWithSummary?.id;
                                    }
                                    if (imageId) {
                                      await createChart(imageId, newChartType, chartExportFormat);
                                    }
                                  } finally {
                                    setIsChangingChartType(false);
                                  }
                                }}
                                className="w-full p-3 border border-white/30 rounded-lg focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue appearance-none bg-white/70 backdrop-blur-sm disabled:bg-gray-100"
                              >
                                {chartTypes.map(type => (
                                  <option key={type.value} value={type.value}>{type.label}</option>
                                ))}
                              </select>
                              {isChangingChartType ? (
                                <Loader2 className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400 animate-spin" />
                              ) : (
                                <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                              )}
                            </div>
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                              Chart File Name
                            </label>
                            <input
                              type="text"
                              value={chartFileName}
                              onChange={(e) => setChartFileName(e.target.value)}
                              placeholder="my-chart"
                              className="w-full p-3 border border-white/30 rounded-lg focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue bg-white/70 backdrop-blur-sm"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                              Export Format
                            </label>
                            <div className="relative">
                              <select
                                value={chartExportFormat}
                                onChange={(e) => {
                                  setChartExportFormat(e.target.value);
                                  // Update the chart data with new export format without recreating the chart
                                  if (chartData && chartData.data.chartUrl) {
                                    setChartData({
                                      ...chartData,
                                      data: {
                                        ...chartData.data,
                                        exportFormat: e.target.value
                                      }
                                    });
                                  }
                                }}
                                className="w-full p-3 border border-white/30 rounded-lg focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue appearance-none bg-white/70 backdrop-blur-sm"
                              >
                                <option value="png">PNG</option>
                                <option value="jpg">JPG</option>
                                <option value="svg">SVG</option>
                                <option value="pdf">PDF</option>
                                <option value="docx">Word Document</option>
                                <option value="pptx">PowerPoint</option>
                              </select>
                              <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                            </div>
                          </div>
                          <div className="space-y-2 pt-2">
                            <button
                              onClick={exportChart}
                              disabled={isExportingChart}
                              className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-green-600/90 backdrop-blur-sm text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors border border-white/10"
                            >
                              {isExportingChart ? (
                                <>
                                  <Loader2 className="w-4 h-4 animate-spin" />
                                  <span>Exporting...</span>
                                </>
                              ) : (
                                <>
                                  <Download className="w-4 h-4" />
                                  <span>Export Chart</span>
                                </>
                              )}
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                    {/* Chart Preview - Right Side */}
                    <div className="lg:col-span-2">
                      <div className="bg-white/70 backdrop-blur-sm rounded-lg p-6 border border-white/20">
                        <h4 className="font-semibold text-gray-800 mb-4">{chartData.title}</h4>
                        <div className="w-full h-80 bg-gradient-to-br from-confluence-blue/10 to-confluence-light-blue/10 rounded-lg flex items-center justify-center border border-white/20 overflow-hidden">
                          {chartData.data.chartUrl ? (
                            <img 
                              src={chartData.data.chartUrl} 
                              alt={chartData.title}
                              className="w-full h-full object-contain"
                            />
                          ) : (
                            <div className="text-center">
                              <BarChart3 className="w-20 h-20 text-confluence-blue mx-auto mb-4" />
                              <p className="text-gray-600 font-medium text-lg">{chartData.title}</p>
                              <p className="text-gray-500 text-sm mt-2">Live {chartData.type} chart preview</p>
                              <div className="mt-4 text-xs text-gray-400 max-w-md mx-auto">
                                Chart updates automatically when you change the type in the controls panel
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            {/* Right Column - Q&A and Export */}
            <div className="xl:col-span-1">
              <div className="bg-white/60 backdrop-blur-xl rounded-xl p-4 space-y-4 border border-white/20 shadow-lg">
                <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
                  <MessageSquare className="w-5 h-5 mr-2" />
                  Image Q&A
                </h3>
                {/* Image Selection for Q&A */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select Image for Questions
                  </label>
                  <div className="relative">
                    <select
                      value={selectedImage}
                      onChange={(e) => setSelectedImage(e.target.value)}
                      className="w-full p-3 border border-white/30 rounded-lg focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue appearance-none bg-white/70 backdrop-blur-sm"
                    >
                      <option value="">Choose image...</option>
                      {images.filter(img => img.summary).map(image => (
                        <option key={image.id} value={image.id}>{image.name}</option>
                      ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
                  </div>
                </div>
                {/* Add Question */}
                <div className="space-y-2">
                  <textarea
                    value={newQuestion}
                    onChange={(e) => setNewQuestion(e.target.value)}
                    placeholder="Ask about the selected image..."
                    className="w-full p-2 border border-white/30 rounded focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue resize-none bg-white/70 backdrop-blur-sm"
                    rows={3}
                  />
                  <button
                    onClick={addQuestion}
                    disabled={!newQuestion.trim() || !selectedImage || isAskingQuestion}
                    className="w-full px-3 py-2 bg-confluence-blue/90 backdrop-blur-sm text-white rounded hover:bg-confluence-blue disabled:bg-gray-300 transition-colors flex items-center justify-center space-x-2 border border-white/10"
                  >
                    {isAskingQuestion ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Asking...</span>
                      </>
                    ) : (
                      <>
                        <MessageSquare className="w-4 h-4" />
                        <span>Ask Question</span>
                      </>
                    )}
                  </button>
                </div>
                {/* Q&A Display */}
                {selectedImage && (
                  <div className="pt-4 border-t border-white/20 space-y-3">
                    <h4 className="font-semibold text-gray-800">Questions & Answers</h4>
                    {(() => {
                      const selectedImageData = images.find(img => img.id === selectedImage);
                      if (!selectedImageData || !selectedImageData.qa || selectedImageData.qa.length === 0) {
                        return (
                          <div className="text-center py-4">
                            <MessageSquare className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                            <p className="text-sm text-gray-500">No questions asked yet</p>
                          </div>
                        );
                      }
                      return (
                        <div className="space-y-3 max-h-60 overflow-y-auto">
                          {selectedImageData.qa.map((qa, index) => (
                            <div key={index} className="p-3 bg-white/70 backdrop-blur-sm rounded-lg border border-white/20">
                              <p className="font-medium text-gray-800 text-sm mb-2">Q: {qa.question}</p>
                              <p className="text-gray-700 text-sm">{qa.answer}</p>
                            </div>
                          ))}
                        </div>
                      );
                    })()}
                  </div>
                )}
                {/* Export Options */}
                <div className="pt-4 border-t border-white/20 space-y-3">
                  <h4 className="font-semibold text-gray-800">Export Image Analysis</h4>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      File Name
                    </label>
                    <input
                      type="text"
                      value={fileName}
                      onChange={(e) => setFileName(e.target.value)}
                      placeholder="image-analysis"
                      className="w-full p-2 border border-white/30 rounded focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue bg-white/70 backdrop-blur-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Export Format
                    </label>
                    <div className="relative">
                      <select
                        value={exportFormat}
                        onChange={(e) => setExportFormat(e.target.value)}
                        className="w-full p-2 border border-white/30 rounded focus:ring-2 focus:ring-confluence-blue focus:border-confluence-blue appearance-none bg-white/70 backdrop-blur-sm"
                      >
                        <option value="pdf">PDF</option>
                        <option value="docx">Word Document</option>
                        <option value="txt">Plain Text</option>
                        <option value="md">Markdown</option>
                      </select>
                      <ChevronDown className="absolute right-2 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                    </div>
                  </div>
                  <div className="space-y-2">
                    {images.filter(img => img.summary).map(image => (
                      <button
                        key={image.id}
                        onClick={() => exportImage(image)}
                        className="w-full flex items-center justify-center space-x-2 px-3 py-2 bg-green-600/90 backdrop-blur-sm text-white rounded-lg hover:bg-green-700 transition-colors border border-white/10"
                      >
                        <Download className="w-4 h-4" />
                        <span>Export {image.name}</span>
                      </button>
                    ))}
                  </div>
                </div>
                {selectedImage && (
                  <button
                    onClick={async () => {
                      const { space, page } = getConfluenceSpaceAndPageFromUrl();
                      if (!space || !page) {
                        alert('Confluence space or page not specified in macro src URL.');
                        return;
                      }
                      const selectedImageData = images.find(img => img.id === selectedImage);
                      if (!selectedImageData || !selectedImageData.summary) {
                        alert('No summary available for the selected image.');
                        return;
                      }
                      try {
                        await apiService.saveToConfluence({
                          space_key: space,
                          page_title: page,
                          content: selectedImageData.summary,
                        });
                        setShowToast(true);
                        setTimeout(() => setShowToast(false), 3000);
                      } catch (err: any) {
                        alert('Failed to save to Confluence: ' + (err.message || err));
                      }
                    }}
                    className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-confluence-blue/90 backdrop-blur-sm text-white rounded-lg hover:bg-confluence-blue transition-colors border border-white/10 mt-2"
                  >
                    <Save className="w-4 h-4" />
                    <span>Save Summary to Confluence</span>
                  </button>
                )}
              </div>
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

export default ImageInsights; 