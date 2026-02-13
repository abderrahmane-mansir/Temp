import { useState, useEffect } from 'react';
import { Calendar, Search, TrendingUp, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { getBestDays, type BestDayAnalysis } from '@/lib/api';

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

interface BestDayResult {
  platform: string;
  content_type: string;
  best_day: string;
  best_day_number: number;
  viral_rate: number;
  post_count: number;
}

export function BestDayForm() {
  const [platform, setPlatform] = useState<string>('');
  const [contentType, setContentType] = useState<string>('');
  const [region, setRegion] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [analysis, setAnalysis] = useState<BestDayAnalysis | null>(null);
  const [result, setResult] = useState<BestDayResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);

  // Load analysis data on mount
  useEffect(() => {
    const loadAnalysis = async () => {
      try {
        const response = await getBestDays();
        if (response.success && response.analysis) {
          setAnalysis(response.analysis);
        } else {
          setError('No analysis data available. Please train a model first.');
        }
      } catch {
        setError('Failed to load analysis. Please train a model first.');
      } finally {
        setIsLoading(false);
      }
    };
    loadAnalysis();
  }, []);

  const handleSearch = () => {
    if (!analysis || !platform || !contentType) {
      setError('Please select Platform and Content Type');
      return;
    }

    setSearched(true);
    setError(null);

    // Find the best day for selected combination
    const match = analysis.best_days_by_combination?.find(
      (item) =>
        item.platform.toLowerCase() === platform.toLowerCase() &&
        item.content_type.toLowerCase() === contentType.toLowerCase()
    );

    if (match) {
      setResult(match);
    } else {
      setResult(null);
      setError(`No data found for ${platform} / ${contentType}. Try a different combination.`);
    }
  };

  const handleReset = () => {
    setPlatform('');
    setContentType('');
    setRegion('');
    setResult(null);
    setSearched(false);
    setError(null);
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-12">
          <div className="flex flex-col items-center justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4" />
            <p className="text-muted-foreground">Loading analysis data...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Find Best Day to Post</h2>
        <p className="text-muted-foreground">
          Select your platform and content type to discover the optimal posting day
        </p>
      </div>

      {/* Input Form */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5 text-primary" />
            Select Your Options
          </CardTitle>
          <CardDescription>
            Choose your platform and content type to get personalized recommendations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid md:grid-cols-3 gap-4">
            {/* Platform Select */}
            <div className="space-y-2">
              <Label htmlFor="platform">Platform *</Label>
              <select
                id="platform"
                className="w-full px-3 py-2 rounded-md border border-input bg-background text-foreground"
                value={platform}
                onChange={(e) => setPlatform(e.target.value)}
              >
                <option value="">Select Platform</option>
                {analysis?.platforms?.map((p) => (
                  <option key={p} value={p}>
                    {p.charAt(0).toUpperCase() + p.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            {/* Content Type Select */}
            <div className="space-y-2">
              <Label htmlFor="content-type">Content Type *</Label>
              <select
                id="content-type"
                className="w-full px-3 py-2 rounded-md border border-input bg-background text-foreground"
                value={contentType}
                onChange={(e) => setContentType(e.target.value)}
              >
                <option value="">Select Content Type</option>
                {analysis?.content_types?.map((ct) => (
                  <option key={ct} value={ct}>
                    {ct.charAt(0).toUpperCase() + ct.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            {/* Region Select (optional) */}
            <div className="space-y-2">
              <Label htmlFor="region">Region (Optional)</Label>
              <select
                id="region"
                className="w-full px-3 py-2 rounded-md border border-input bg-background text-foreground"
                value={region}
                onChange={(e) => setRegion(e.target.value)}
              >
                <option value="">All Regions</option>
                <option value="usa">USA</option>
                <option value="brazil">Brazil</option>
                <option value="india">India</option>
                <option value="uk">UK</option>
              </select>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 flex items-center gap-3">
              <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0" />
              <p className="text-destructive text-sm">{error}</p>
            </div>
          )}

          {/* Buttons */}
          <div className="flex gap-3">
            <Button onClick={handleSearch} disabled={!platform || !contentType} className="flex-1">
              <Calendar className="h-4 w-4 mr-2" />
              Find Best Day
            </Button>
            <Button variant="outline" onClick={handleReset}>
              Reset
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Result Card */}
      {searched && result && (
        <Card className="border-primary/30 bg-gradient-to-br from-primary/5 to-transparent">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Recommended Best Day
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Main Result */}
            <div className="text-center py-6">
              <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-primary/10 mb-4">
                <Calendar className="h-10 w-10 text-primary" />
              </div>
              <h3 className="text-4xl font-bold text-primary mb-2">{result.best_day}</h3>
              <p className="text-muted-foreground">
                Best day to post <span className="font-medium text-foreground">{result.content_type}</span> on{' '}
                <span className="font-medium text-foreground capitalize">{result.platform}</span>
              </p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-secondary/50 rounded-lg p-4 text-center">
                <p className="text-3xl font-bold text-green-500">
                  {(result.viral_rate * 100).toFixed(1)}%
                </p>
                <p className="text-sm text-muted-foreground">Viral Rate</p>
              </div>
              <div className="bg-secondary/50 rounded-lg p-4 text-center">
                <p className="text-3xl font-bold text-foreground">
                  {result.post_count}
                </p>
                <p className="text-sm text-muted-foreground">Posts Analyzed</p>
              </div>
            </div>

            {/* All Days Comparison */}
            <div>
              <h4 className="font-medium mb-3">All Days Comparison</h4>
              <div className="flex gap-1">
                {DAYS.map((day, index) => {
                  const dayData = analysis?.heatmap_data?.find(
                    (h) =>
                      h.platform === result.platform &&
                      h.content_type === result.content_type &&
                      h.day_number === index
                  );
                  const rate = dayData?.viral_rate || 0;
                  const isBest = day === result.best_day;

                  return (
                    <div
                      key={day}
                      className={`flex-1 rounded-lg p-2 text-center transition-all ${
                        isBest
                          ? 'bg-primary text-primary-foreground ring-2 ring-primary ring-offset-2 ring-offset-background'
                          : 'bg-secondary/50'
                      }`}
                    >
                      <p className="text-xs font-medium">{day.slice(0, 3)}</p>
                      <p className={`text-sm font-bold ${isBest ? '' : 'text-muted-foreground'}`}>
                        {dayData ? `${(rate * 100).toFixed(0)}%` : '-'}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Tip */}
            <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
              <p className="text-sm text-blue-400">
                💡 <strong>Tip:</strong> Post your {result.content_type} content on {result.platform} every{' '}
                {result.best_day} to maximize your chances of going viral!
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* No Result Found */}
      {searched && !result && !error && (
        <Card>
          <CardContent className="py-8 text-center">
            <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-lg font-medium">No data available</p>
            <p className="text-muted-foreground">
              Try selecting a different platform or content type combination
            </p>
          </CardContent>
        </Card>
      )}

      {/* Overall Stats */}
      {analysis && !searched && (
        <Card>
          <CardHeader>
            <CardTitle>Quick Stats</CardTitle>
            <CardDescription>Overview from your training data</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-secondary/50 rounded-lg p-4 text-center">
                <p className="text-2xl font-bold text-primary">{analysis.overall_best_day}</p>
                <p className="text-sm text-muted-foreground">Overall Best Day</p>
              </div>
              <div className="bg-secondary/50 rounded-lg p-4 text-center">
                <p className="text-2xl font-bold text-foreground">
                  {analysis.total_posts_analyzed?.toLocaleString()}
                </p>
                <p className="text-sm text-muted-foreground">Posts Analyzed</p>
              </div>
              <div className="bg-secondary/50 rounded-lg p-4 text-center">
                <p className="text-2xl font-bold text-foreground">
                  {analysis.best_days_by_combination?.length}
                </p>
                <p className="text-sm text-muted-foreground">Combinations</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
