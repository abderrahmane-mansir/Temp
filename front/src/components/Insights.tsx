import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, ScatterChart, Scatter } from 'recharts';
import { TrendingUp, Users, Calendar, Hash, BarChart3, PieChart as PieChartIcon } from 'lucide-react';
// import { getInsightsData, type InsightsData } from '../lib/api';

const Insights: React.FC = () => {
  // Static fake data for display
  const data = {
    totalPosts: 4500,
    viralPosts: 1125,
    viralRate: 0.25,
    bestHashtag: '#dance',
    bestHashtagRate: 0.34,
    bestDay: 'Sunday',
    bestDayRate: 0.38,
    platformData: [
      { name: 'Youtube', posts: 1250, viralRate: 0.23 },
      { name: 'Twitter', posts: 1180, viralRate: 0.28 },
      { name: 'Instagram', posts: 1070, viralRate: 0.24 },
      { name: 'Tiktok', posts: 1000, viralRate: 0.31 }
    ],
    contentTypeData: [
      { name: 'Video', posts: 1800, viralRate: 0.31, color: '#8884d8' },
      { name: 'Image', posts: 1200, viralRate: 0.18, color: '#82ca9d' },
      { name: 'Reel', posts: 900, viralRate: 0.29, color: '#ffc658' },
      { name: 'Text', posts: 600, viralRate: 0.22, color: '#ff7300' }
    ],
    hashtagData: [
      { hashtag: '#dance', posts: 520, viralRate: 0.34 },
      { hashtag: '#music', posts: 480, viralRate: 0.29 },
      { hashtag: '#comedy', posts: 410, viralRate: 0.26 },
      { hashtag: '#fitness', posts: 380, viralRate: 0.18 },
      { hashtag: '#food', posts: 290, viralRate: 0.22 }
    ],
    dayOfWeekData: [
      { day: 'Mon', dayNum: 0, avgViralRate: 0.18 },
      { day: 'Tue', dayNum: 1, avgViralRate: 0.22 },
      { day: 'Wed', dayNum: 2, avgViralRate: 0.25 },
      { day: 'Thu', dayNum: 3, avgViralRate: 0.28 },
      { day: 'Fri', dayNum: 4, avgViralRate: 0.32 },
      { day: 'Sat', dayNum: 5, avgViralRate: 0.35 },
      { day: 'Sun', dayNum: 6, avgViralRate: 0.38 }
    ],
    regionData: [
      { name: 'USA', value: 45, viral: 25 },
      { name: 'BRAZIL', value: 35, viral: 28 },
      { name: 'UK', value: 20, viral: 22 }
    ],
    correlationData: [
      { feature: 'Virality Score', correlation: 0.82 },
      { feature: 'Shares', correlation: 0.78 },
      { feature: 'Engagement Rate', correlation: 0.71 },
      { feature: 'Comments', correlation: 0.65 },
      { feature: 'Likes', correlation: 0.52 },
      { feature: 'Views', correlation: 0.43 }
    ],
    bestDayHeatmap: [
      { platform: 'Youtube', contentType: 'Video', bestDay: 'Friday', viralRate: 0.41 },
      { platform: 'Twitter', contentType: 'Text', bestDay: 'Wednesday', viralRate: 0.33 },
      { platform: 'Instagram', contentType: 'Reel', bestDay: 'Saturday', viralRate: 0.45 },
      { platform: 'Tiktok', contentType: 'Video', bestDay: 'Sunday', viralRate: 0.38 },
      { platform: 'Instagram', contentType: 'Image', bestDay: 'Sunday', viralRate: 0.29 },
      { platform: 'Youtube', contentType: 'Image', bestDay: 'Thursday', viralRate: 0.28 }
    ]
  };

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00c49f'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2 mb-6">
        <BarChart3 className="h-6 w-6 text-blue-600" />
        <h1 className="text-2xl font-bold text-gray-900">Data Insights</h1>
        <Badge variant="secondary" className="ml-2">Training Dataset Analysis</Badge>
      </div>

      {/* Key Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Posts</p>
                <p className="text-2xl font-bold text-blue-600">{data.totalPosts.toLocaleString()}</p>
              </div>
              <Users className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Viral Posts</p>
                <p className="text-2xl font-bold text-green-600">{data.viralPosts.toLocaleString()}</p>
                <p className="text-xs text-gray-500">{(data.viralRate * 100).toFixed(1)}% viral rate</p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Best Day</p>
                <p className="text-2xl font-bold text-purple-600">{data.bestDay}</p>
                <p className="text-xs text-gray-500">{(data.bestDayRate * 100).toFixed(1)}% viral rate</p>
              </div>
              <Calendar className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Top Hashtag</p>
                <p className="text-2xl font-bold text-orange-600">{data.bestHashtag}</p>
                <p className="text-xs text-gray-500">{(data.bestHashtagRate * 100).toFixed(1)}% viral rate</p>
              </div>
              <Hash className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Platform Performance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Platform Performance Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.platformData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip 
                  formatter={(value, name) => [
                    name === 'posts' ? `${value} posts` : `${(value * 100).toFixed(1)}%`,
                    name === 'posts' ? 'Total Posts' : 'Viral Rate'
                  ]}
                />
                <Bar yAxisId="left" dataKey="posts" fill="#8884d8" name="posts" />
                <Bar yAxisId="right" dataKey="viralRate" fill="#82ca9d" name="viralRate" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 text-sm text-gray-600">
            <strong>Insight:</strong> {data.platformData.length > 0 && 
              `${data.platformData.reduce((max, p) => p.viralRate > max.viralRate ? p : max).name} shows the highest viral rate.`
            }
          </div>
        </CardContent>
      </Card>

      {/* Content Type Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PieChartIcon className="h-5 w-5" />
              Content Type Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={data.contentTypeData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="posts"
                  >
                    {data.contentTypeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`${value} posts`, 'Count']} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-2 text-sm text-gray-600">
              <strong>Key Finding:</strong> {data.contentTypeData.length > 0 && 
                `${data.contentTypeData.reduce((max, c) => c.viralRate > max.viralRate ? c : max).name} content has the highest viral rate (${(data.contentTypeData.reduce((max, c) => c.viralRate > max.viralRate ? c : max).viralRate * 100).toFixed(1)}%).`
              }
            </div>
          </CardContent>
        </Card>

      </div>

      {/* Best Day Heatmap */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            Optimal Posting Schedule by Platform & Content
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2 font-semibold">Platform</th>
                  <th className="text-left p-2 font-semibold">Content Type</th>
                  <th className="text-left p-2 font-semibold">Best Day</th>
                  <th className="text-left p-2 font-semibold">Viral Rate</th>
                </tr>
              </thead>
              <tbody>
                {data.bestDayHeatmap.map((item, index) => (
                  <tr key={index} className="border-b hover:bg-gray-50">
                    <td className="p-2">
                      <Badge variant="outline">{item.platform}</Badge>
                    </td>
                    <td className="p-2">{item.contentType}</td>
                    <td className="p-2 font-semibold text-blue-600">{item.bestDay}</td>
                    <td className="p-2">
                      <div className="flex items-center gap-2">
                        <div 
                          className="h-2 bg-green-500 rounded"
                          style={{ width: `${item.viralRate * 100}px` }}
                        />
                        <span className="text-sm">{(item.viralRate * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 text-sm text-gray-600">
            <strong>Strategic Insight:</strong> Instagram Reels perform best on Saturdays (45% viral rate), while YouTube Videos peak on Fridays (41%).
          </div>
        </CardContent>
      </Card>

      {/* Regional Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Regional Performance Distribution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {data.regionData.map((region, index) => (
              <div key={region.name} className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{region.name}</div>
                <div className="text-sm text-gray-600 mt-1">{region.value}% of posts</div>
                <div className="text-sm text-green-600 font-semibold">{region.viral}% viral rate</div>
                <div 
                  className="mt-2 h-2 bg-blue-200 rounded-full overflow-hidden"
                >
                  <div 
                    className="h-full bg-blue-500 transition-all duration-300"
                    style={{ width: `${Math.min(region.viral * 2, 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 text-sm text-gray-600">
            <strong>Geographic Insight:</strong> {data.regionData.length > 0 && 
              `${data.regionData.reduce((max, r) => r.viral > max.viral ? r : max).name} shows the highest viral rate relative to post volume.`
            }
          </div>
        </CardContent>
      </Card>

      {/* Key Takeaways */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-blue-700">
            <TrendingUp className="h-5 w-5" />
            Key Strategic Takeaways
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Content Strategy</h4>
              <ul className="space-y-1 text-sm text-gray-700">
                <li>• Focus on video content (31% viral rate)</li>
                <li>• Use #dance and #music hashtags</li>
                <li>• Prioritize engagement over views</li>
                <li>• Target shares and comments</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Timing Strategy</h4>
              <ul className="space-y-1 text-sm text-gray-700">
                <li>• Post on weekends (35-38% viral rate)</li>
                <li>• Sunday is optimal across platforms</li>
                <li>• Instagram Reels: Saturday</li>
                <li>• YouTube Videos: Friday</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Insights;