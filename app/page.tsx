"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Loader2, AlertTriangle, CheckCircle, BarChart3, Database, Brain, Shield, Target } from "lucide-react"

export default function FakeNewsDetector() {
  const [headline, setHeadline] = useState("")
  const [content, setContent] = useState("")
  const [prediction, setPrediction] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [modelStats, setModelStats] = useState<any>(null)

  const handlePredict = async () => {
    if (!headline.trim() && !content.trim()) {
      alert("Please enter either a headline or content to analyze")
      return
    }

    setLoading(true)
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          headline: headline.trim(),
          content: content.trim(),
        }),
      })

      const result = await response.json()
      setPrediction(result)
    } catch (error) {
      console.error("Error:", error)
      alert("Error making prediction. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const handleTrainModel = async () => {
    setLoading(true)
    try {
      const response = await fetch("/api/train", {
        method: "POST",
      })
      const result = await response.json()
      setModelStats(result)
      alert("Model trained successfully!")
    } catch (error) {
      console.error("Error:", error)
      alert("Error training model. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const handleProcessData = async () => {
    setLoading(true)
    try {
      const response = await fetch("/api/process-data", {
        method: "POST",
      })
      const result = await response.json()
      alert("Data processed successfully!")
    } catch (error) {
      console.error("Error:", error)
      alert("Error processing data. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const testExamples = [
    {
      type: "fake",
      headline: "SHOCKING: Scientists Discover Aliens Living Among Us - Government Cover-Up Exposed!",
      content:
        "In an unbelievable turn of events, unnamed sources have leaked classified documents proving that extraterrestrial beings have been secretly living in major cities worldwide. This explosive revelation will change everything you thought you knew about reality. The mainstream media is trying to suppress this story, but the truth cannot be hidden any longer!",
    },
    {
      type: "real",
      headline: "New Study Shows Regular Exercise Reduces Risk of Heart Disease by 30%",
      content:
        "According to a comprehensive study published in the Journal of the American Medical Association, researchers at Harvard Medical School found that individuals who engage in regular physical activity have a significantly lower risk of developing cardiovascular disease. The study, which followed 50,000 participants over 10 years, provides strong evidence for the health benefits of exercise.",
    },
  ]

  const loadExample = (example: any) => {
    setHeadline(example.headline)
    setContent(example.content)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-emerald-100 p-4">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-3">
            <Shield className="h-10 w-10 text-purple-600" />
            <h1 className="text-4xl font-bold text-slate-900">Fake News Detection System</h1>
          </div>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Advanced AI-powered system to detect fake news using Natural Language Processing and Machine Learning
          </p>
        </div>

        {/* Test Examples */}
        <Card className="border-purple-200 bg-white/70 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-purple-700">
              <Target className="h-5 w-5" />
              Quick Test Examples
            </CardTitle>
            <CardDescription>Try these examples to see how the model works</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {testExamples.map((example, index) => (
                <div key={index} className="p-4 border rounded-lg bg-white/50">
                  <div className="flex items-center justify-between mb-2">
                    <Badge variant={example.type === "fake" ? "destructive" : "default"}>
                      {example.type === "fake" ? "Fake Example" : "Real Example"}
                    </Badge>
                    <Button size="sm" variant="outline" onClick={() => loadExample(example)} className="text-xs">
                      Load Example
                    </Button>
                  </div>
                  <p className="text-sm text-slate-600 line-clamp-2">{example.headline}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Input Section */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="border-emerald-200 bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-emerald-700">
                  <Brain className="h-5 w-5" />
                  News Article Analysis
                </CardTitle>
                <CardDescription>Enter a news headline and/or content to analyze for authenticity</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium mb-2 block text-slate-700">News Headline</label>
                  <Input
                    placeholder="Enter news headline..."
                    value={headline}
                    onChange={(e) => setHeadline(e.target.value)}
                    className="border-emerald-200 focus:border-emerald-400"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block text-slate-700">News Content</label>
                  <Textarea
                    placeholder="Enter news article content..."
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    rows={8}
                    className="border-emerald-200 focus:border-emerald-400"
                  />
                </div>
                <Button
                  onClick={handlePredict}
                  disabled={loading || (!headline.trim() && !content.trim())}
                  className="w-full bg-gradient-to-r from-purple-600 to-emerald-600 hover:from-purple-700 hover:to-emerald-700"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze News Article"
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Prediction Results */}
            {prediction && (
              <Card className="border-slate-200 bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    {prediction.prediction === "REAL" ? (
                      <CheckCircle className="h-5 w-5 text-emerald-500" />
                    ) : (
                      <AlertTriangle className="h-5 w-5 text-red-500" />
                    )}
                    Analysis Results
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Classification:</span>
                    <Badge
                      variant={prediction.prediction === "REAL" ? "default" : "destructive"}
                      className={prediction.prediction === "REAL" ? "bg-emerald-500" : ""}
                    >
                      {prediction.prediction === "REAL" ? "Real News" : "Fake News"}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Confidence:</span>
                    <span className="text-sm font-bold">{(prediction.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all duration-500 ${
                        prediction.prediction === "REAL"
                          ? "bg-gradient-to-r from-emerald-400 to-emerald-600"
                          : "bg-gradient-to-r from-red-400 to-red-600"
                      }`}
                      style={{ width: `${prediction.confidence * 100}%` }}
                    ></div>
                  </div>

                  {/* Detailed Analysis */}
                  {prediction.analysis && (
                    <div className="mt-4 p-4 bg-slate-50 rounded-lg">
                      <h4 className="font-medium text-slate-700 mb-2">Detailed Analysis:</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span>Fake Indicators:</span>
                          <span className="font-medium text-red-600">{prediction.analysis.fake_score}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Real Indicators:</span>
                          <span className="font-medium text-emerald-600">{prediction.analysis.real_score}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Sentiment Score:</span>
                          <span className="font-medium">{prediction.analysis.sentiment_score?.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Text Length:</span>
                          <span className="font-medium">{prediction.analysis.text_length}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>

          {/* Control Panel */}
          <div className="space-y-6">
            <Card className="border-purple-200 bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-purple-700">
                  <BarChart3 className="h-5 w-5" />
                  Model Controls
                </CardTitle>
                <CardDescription>Manage data processing and model training</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button
                  onClick={handleProcessData}
                  disabled={loading}
                  variant="outline"
                  className="w-full border-purple-200 hover:bg-purple-50 bg-transparent"
                >
                  <Database className="mr-2 h-4 w-4" />
                  Process Training Data
                </Button>
                <Button
                  onClick={handleTrainModel}
                  disabled={loading}
                  variant="outline"
                  className="w-full border-emerald-200 hover:bg-emerald-50 bg-transparent"
                >
                  <Brain className="mr-2 h-4 w-4" />
                  Train Model
                </Button>
              </CardContent>
            </Card>

            {/* Model Statistics */}
            {modelStats && (
              <Card className="border-slate-200 bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-slate-700">Model Performance</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm">Accuracy:</span>
                    <span className="font-bold text-emerald-600">{(modelStats.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Precision:</span>
                    <span className="font-bold text-purple-600">{(modelStats.precision * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Recall:</span>
                    <span className="font-bold text-emerald-600">{(modelStats.recall * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">F1-Score:</span>
                    <span className="font-bold text-purple-600">{(modelStats.f1_score * 100).toFixed(1)}%</span>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Information Panel */}
            <Card className="border-slate-200 bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-slate-700">How It Works</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="space-y-2">
                  <h4 className="font-medium text-purple-700">1. Advanced Text Analysis</h4>
                  <p className="text-slate-600">Analyzes linguistic patterns, sentiment, and structural features</p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium text-emerald-700">2. Multi-Factor Scoring</h4>
                  <p className="text-slate-600">Combines multiple indicators for accurate classification</p>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium text-purple-700">3. Machine Learning</h4>
                  <p className="text-slate-600">Uses trained models to detect fake news patterns</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
