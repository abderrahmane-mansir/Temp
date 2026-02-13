import { useState } from 'react';
import { Send, TrendingUp, TrendingDown, Sparkles, ArrowUp, ArrowDown, Info } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { explainPrediction, type PredictRequest, type ExplainedPrediction, type ShapContribution } from '@/lib/api';

const platformOptions = [
  { value: 'tiktok', label: 'TikTok' },
  { value: 'youtube', label: 'YouTube' },
  { value: 'instagram', label: 'Instagram' },
  { value: 'twitter', label: 'Twitter' },
  { value: 'facebook', label: 'Facebook' },
];

const contentTypeOptions = [
  { value: 'video', label: 'Video' },
  { value: 'reel', label: 'Reel' },
  { value: 'story', label: 'Story' },
  { value: 'post', label: 'Post' },
  { value: 'image', label: 'Image' },
];

const regionOptions = [
  { value: 'usa', label: 'USA' },
  { value: 'uk', label: 'UK' },
  { value: 'india', label: 'India' },
  { value: 'brazil', label: 'Brazil' },
  { value: 'germany', label: 'Germany' },
  { value: 'france', label: 'France' },
  { value: 'japan', label: 'Japan' },
  { value: 'canada', label: 'Canada' },
];

export function PredictForm() {
  const [formData, setFormData] = useState<PredictRequest>({
    Post_ID: '',
    Pseudo_Caption: '',
    Post_Date: new Date().toISOString().split('T')[0],
    Platform: 'tiktok',
    Hashtag: '',
    Content_Type: 'video',
    Region: 'usa',
    Views: 0,
    Likes: 0,
    Shares: 0,
    Comments: 0,
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExplainedPrediction | null>(null);

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target as HTMLInputElement;
    setFormData((prev: PredictRequest) => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // Auto-generate caption if not provided
      const dataToSend = {
        ...formData,
        Pseudo_Caption:
          formData.Pseudo_Caption ||
          `${formData.Platform} ${formData.Content_Type} about ${formData.Hashtag || '#content'} posted in ${formData.Region}`,
        Post_ID: formData.Post_ID || `POST_${Date.now()}`,
      };

      const response = await explainPrediction(dataToSend);
      if (response.success && response.predictions.length > 0) {
        setResult(response.predictions[0]);
      } else {
        setError(response.error || 'Prediction failed');
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get prediction';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Input Form */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Send className="h-5 w-5 text-primary" />
            Predict Virality
          </CardTitle>
          <CardDescription>
            Enter post details to predict if it will go viral
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Platform and Content Type */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="Platform">Platform</Label>
                <Select
                  id="Platform"
                  name="Platform"
                  value={formData.Platform}
                  onChange={handleInputChange}
                  options={platformOptions}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="Content_Type">Content Type</Label>
                <Select
                  id="Content_Type"
                  name="Content_Type"
                  value={formData.Content_Type}
                  onChange={handleInputChange}
                  options={contentTypeOptions}
                />
              </div>
            </div>

            {/* Region and Date */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="Region">Region</Label>
                <Select
                  id="Region"
                  name="Region"
                  value={formData.Region}
                  onChange={handleInputChange}
                  options={regionOptions}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="Post_Date">Post Date</Label>
                <Input
                  id="Post_Date"
                  name="Post_Date"
                  type="date"
                  value={formData.Post_Date}
                  onChange={handleInputChange}
                />
              </div>
            </div>

            {/* Hashtag */}
            <div className="space-y-2">
              <Label htmlFor="Hashtag">Hashtag</Label>
              <Input
                id="Hashtag"
                name="Hashtag"
                placeholder="#dance, #comedy, #fitness..."
                value={formData.Hashtag}
                onChange={handleInputChange}
              />
            </div>

            {/* Engagement Metrics */}
            <div className="space-y-3">
              <Label className="text-base">Engagement Metrics</Label>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="Views" className="text-muted-foreground text-xs">
                    Views
                  </Label>
                  <Input
                    id="Views"
                    name="Views"
                    type="number"
                    min="0"
                    value={formData.Views}
                    onChange={handleInputChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="Likes" className="text-muted-foreground text-xs">
                    Likes
                  </Label>
                  <Input
                    id="Likes"
                    name="Likes"
                    type="number"
                    min="0"
                    value={formData.Likes}
                    onChange={handleInputChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="Shares" className="text-muted-foreground text-xs">
                    Shares
                  </Label>
                  <Input
                    id="Shares"
                    name="Shares"
                    type="number"
                    min="0"
                    value={formData.Shares}
                    onChange={handleInputChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="Comments" className="text-muted-foreground text-xs">
                    Comments
                  </Label>
                  <Input
                    id="Comments"
                    name="Comments"
                    type="number"
                    min="0"
                    value={formData.Comments}
                    onChange={handleInputChange}
                  />
                </div>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-3 text-destructive text-sm">
                {error}
              </div>
            )}

            {/* Submit Button */}
            <Button type="submit" className="w-full" size="lg" disabled={isLoading}>
              {isLoading ? (
                <>
                  <Sparkles className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Predict Virality
                </>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Results Card */}
      <Card className={result ? 'ring-2 ring-primary/20' : ''}>
        <CardHeader>
          <CardTitle>Prediction Result</CardTitle>
          <CardDescription>
            {result
              ? 'Analysis complete with AI explainability'
              : 'Fill out the form and click predict to see results'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {result ? (
            <div className="space-y-6">
              {/* Main Result */}
              <div className="text-center p-6 rounded-xl bg-secondary/50">
                {result.viral === 1 ? (
                  <>
                    <TrendingUp className="h-16 w-16 text-success mx-auto mb-3" />
                    <Badge variant="success" className="text-lg px-4 py-1">
                      Likely to Go Viral! 🔥
                    </Badge>
                  </>
                ) : (
                  <>
                    <TrendingDown className="h-16 w-16 text-muted-foreground mx-auto mb-3" />
                    <Badge variant="secondary" className="text-lg px-4 py-1">
                      Unlikely to Go Viral
                    </Badge>
                  </>
                )}
              </div>

              {/* Probability Bars */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Viral Probability</span>
                    <span className="font-medium text-success">
                      {(result.probability_viral * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress
                    value={result.probability_viral * 100}
                    variant="success"
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Non-Viral Probability</span>
                    <span className="font-medium">
                      {(result.probability_not_viral * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={result.probability_not_viral * 100} />
                </div>
              </div>

              {/* SHAP Explanation */}
              {result.explanation && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <Info className="h-4 w-4 text-primary" />
                    <h4 className="font-medium">AI Explanation (SHAP)</h4>
                  </div>
                  
                  {/* Top Factors */}
                  <div className="grid grid-cols-2 gap-3">
                    {/* Positive Factors */}
                    <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                      <p className="text-xs font-medium text-green-500 mb-2 flex items-center gap-1">
                        <ArrowUp className="h-3 w-3" />
                        Helping Virality
                      </p>
                      <div className="space-y-2">
                        {result.explanation.top_positive.length > 0 ? (
                          result.explanation.top_positive.map((item: ShapContribution, idx: number) => (
                            <div key={idx} className="text-xs">
                              <div className="flex justify-between">
                                <span className="text-foreground">{item.feature}</span>
                                <span className="text-green-500 font-medium">+{Math.abs(item.shap_value).toFixed(3)}</span>
                              </div>
                            </div>
                          ))
                        ) : (
                          <p className="text-xs text-muted-foreground">No positive factors</p>
                        )}
                      </div>
                    </div>

                    {/* Negative Factors */}
                    <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                      <p className="text-xs font-medium text-red-500 mb-2 flex items-center gap-1">
                        <ArrowDown className="h-3 w-3" />
                        Hurting Virality
                      </p>
                      <div className="space-y-2">
                        {result.explanation.top_negative.length > 0 ? (
                          result.explanation.top_negative.map((item: ShapContribution, idx: number) => (
                            <div key={idx} className="text-xs">
                              <div className="flex justify-between">
                                <span className="text-foreground">{item.feature}</span>
                                <span className="text-red-500 font-medium">{item.shap_value.toFixed(3)}</span>
                              </div>
                            </div>
                          ))
                        ) : (
                          <p className="text-xs text-muted-foreground">No negative factors</p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* All Feature Contributions */}
                  <div className="bg-secondary/30 rounded-lg p-3">
                    <p className="text-xs font-medium mb-3">Feature Contributions</p>
                    <div className="space-y-2">
                      {result.explanation.contributions.slice(0, 6).map((item: ShapContribution, idx: number) => (
                        <div key={idx} className="flex items-center gap-2 text-xs">
                          <span className="w-24 truncate text-muted-foreground">{item.feature}</span>
                          <div className="flex-1 h-4 bg-secondary rounded-full overflow-hidden relative">
                            <div
                              className={`absolute h-full ${item.impact === 'positive' ? 'bg-green-500' : item.impact === 'negative' ? 'bg-red-500' : 'bg-gray-500'}`}
                              style={{
                                width: `${Math.min(Math.abs(item.shap_value) * 100, 100)}%`,
                                left: item.shap_value < 0 ? 'auto' : '50%',
                                right: item.shap_value < 0 ? '50%' : 'auto',
                              }}
                            />
                            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />
                          </div>
                          <span className={`w-16 text-right font-mono ${item.impact === 'positive' ? 'text-green-500' : item.impact === 'negative' ? 'text-red-500' : ''}`}>
                            {item.shap_value > 0 ? '+' : ''}{item.shap_value.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      Base value: {result.explanation.base_value.toFixed(3)}
                    </p>
                  </div>
                </div>
              )}

              {/* Tips */}
              <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
                <h4 className="font-medium text-sm mb-2">💡 Tips to Improve</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  {result.viral === 0 && (
                    <>
                      <li>• Try posting video content for higher engagement</li>
                      <li>• Use trending hashtags relevant to your niche</li>
                      <li>• Optimize posting time for your target region</li>
                      <li>• Aim for higher engagement rates (likes/views)</li>
                    </>
                  )}
                  {result.viral === 1 && (
                    <>
                      <li>• Great engagement metrics! Keep it up</li>
                      <li>• Consider cross-posting to other platforms</li>
                      <li>• Engage with comments to boost visibility</li>
                    </>
                  )}
                </ul>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
              <Sparkles className="h-12 w-12 mb-4 opacity-50" />
              <p>Enter post details to get a prediction</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
