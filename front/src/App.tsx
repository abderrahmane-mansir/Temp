import { useState, useEffect } from 'react';
import {
  Zap,
  FileSpreadsheet,
  Activity,
  TrendingUp,
  Database,
  Calendar,
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { TrainModel } from '@/components/TrainModel';
import { PredictForm } from '@/components/PredictForm';
import { BatchPredict } from '@/components/BatchPredict';
import { BestDayForm } from '@/components/BestDayForm';
import Insights from '@/components/Insights';
import { getModelStatus, type ModelStatus } from '@/lib/api';

type Tab = 'dashboard' | 'train' | 'predict' | 'batch' | 'bestday' | 'insights';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('dashboard');
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchStatus = async () => {
    try {
      const status = await getModelStatus();
      setModelStatus(status);
    } catch {
      setModelStatus(null);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const tabs = [
    { id: 'dashboard' as const, label: 'Dashboard', icon: Activity },
    { id: 'train' as const, label: 'Train Model', icon: Database },
    { id: 'predict' as const, label: 'Single Predict', icon: Zap },
    { id: 'batch' as const, label: 'Batch Predict', icon: FileSpreadsheet },
    { id: 'bestday' as const, label: 'Best Day', icon: Calendar },
    { id: 'insights' as const, label: 'Data Insights', icon: TrendingUp },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary/50 flex items-center justify-center">
                <TrendingUp className="h-5 w-5 text-primary-foreground" />
              </div>
              <div>
                <h1 className="font-bold text-lg">ViralPredict</h1>
                <p className="text-xs text-muted-foreground">ML-Powered Virality Analysis</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="border-b border-border bg-card/30">
        <div className="container mx-auto px-4">
          <div className="flex gap-1 overflow-x-auto">
            {tabs.map((tab) => (
              <Button
                key={tab.id}
                variant={activeTab === tab.id ? 'secondary' : 'ghost'}
                className={`rounded-none border-b-2 px-6 ${
                  activeTab === tab.id
                    ? 'border-primary'
                    : 'border-transparent'
                }`}
                onClick={() => setActiveTab(tab.id)}
              >
                <tab.icon className="mr-2 h-4 w-4" />
                {tab.label}
              </Button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {activeTab === 'dashboard' && (
          <Dashboard
            modelStatus={modelStatus}
            isLoading={isLoading}
            onRefresh={fetchStatus}
            onNavigate={setActiveTab}
          />
        )}
        {activeTab === 'train' && <TrainModel onTrainingComplete={fetchStatus} />}
        {activeTab === 'predict' && <PredictForm />}
        {activeTab === 'batch' && <BatchPredict />}
        {activeTab === 'bestday' && <BestDayForm />}
        {activeTab === 'insights' && <Insights />}
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-6 mt-auto">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>ViralPredict • XGBoost-powered social media virality prediction</p>
        </div>
      </footer>
    </div>
  );
}

interface DashboardProps {
  modelStatus: ModelStatus | null;
  isLoading: boolean;
  onRefresh: () => void;
  onNavigate: (tab: Tab) => void;
}

function Dashboard({ modelStatus, isLoading, onRefresh, onNavigate }: DashboardProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto" />
          <p className="mt-4 text-muted-foreground">Loading model status...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center space-y-4">
        <h2 className="text-4xl font-bold bg-gradient-to-r from-foreground to-muted-foreground bg-clip-text text-transparent">
          Viral Content Prediction
        </h2>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Leverage machine learning to predict which social media posts will go viral.
          Train on your data or analyze individual posts.
        </p>
      </div>

      {/* Status Cards */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Database className="h-5 w-5 text-primary" />
              Quick Stats
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model Type</span>
                <span className="font-medium">XGBoost</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Balancing</span>
                <span className="font-medium">ADASYN</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Activity className="h-5 w-5 text-primary" />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">API</span>
                <Badge variant="success">Online</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Backend</span>
                <Badge variant="success">Connected</Badge>
              </div>
              <Button
                size="sm"
                variant="outline"
                onClick={onRefresh}
                className="w-full mt-2"
              >
                Refresh Status
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>Jump to common tasks</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid sm:grid-cols-4 gap-4">
            <Button
              variant="outline"
              className="h-auto py-6 flex-col gap-2"
              onClick={() => onNavigate('train')}
            >
              <span>Train Model</span>
              <span className="text-xs text-muted-foreground">
                Upload CSV dataset
              </span>
            </Button>
            <Button
              variant="outline"
              className="h-auto py-6 flex-col gap-2"
              onClick={() => onNavigate('predict')}
            >
              <Zap className="h-8 w-8 text-primary" />
              <span>Single Prediction</span>
              <span className="text-xs text-muted-foreground">
                Analyze one post
              </span>
            </Button>
            <Button
              variant="outline"
              className="h-auto py-6 flex-col gap-2"
              onClick={() => onNavigate('batch')}
            >
              <FileSpreadsheet className="h-8 w-8 text-primary" />
              <span>Batch Predictions</span>
              <span className="text-xs text-muted-foreground">
                Process multiple posts
              </span>
            </Button>
            <Button
              variant="outline"
              className="h-auto py-6 flex-col gap-2"
              onClick={() => onNavigate('bestday')}
            >
              <Calendar className="h-8 w-8 text-primary" />
              <span>Best Day Analysis</span>
              <span className="text-xs text-muted-foreground">
                Find optimal posting days
              </span>
            </Button>
            <Button
              variant="outline"
              className="h-auto py-6 flex-col gap-2"
              onClick={() => onNavigate('insights')}
            >
              <TrendingUp className="h-8 w-8 text-primary" />
              <span>Data Insights</span>
              <span className="text-xs text-muted-foreground">
                Training data analysis
              </span>
            </Button>
          </div>
        </CardContent>
      </Card>

  
    </div>
  );
}
