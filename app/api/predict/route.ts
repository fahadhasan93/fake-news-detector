import { type NextRequest, NextResponse } from "next/server"

// Enhanced fake news detection with multiple indicators
export async function POST(request: NextRequest) {
  try {
    const { headline, content } = await request.json()

    // Combine headline and content
    const message = `${headline} ${content}`.trim()

    if (!message) {
      return NextResponse.json({ error: "No text provided" }, { status: 400 })
    }

    // Advanced text preprocessing
    const processedText = message
      .toLowerCase()
      .replace(/[^\w\s]/g, " ") // Replace punctuation with spaces
      .replace(/\s+/g, " ") // Normalize whitespace
      .trim()

    // Enhanced fake news indicators with weights
    const fakeIndicators = {
      // Sensational language
      sensational: [
        "shocking",
        "unbelievable",
        "amazing",
        "incredible",
        "mind-blowing",
        "explosive",
        "bombshell",
        "stunning",
      ],
      // Clickbait phrases
      clickbait: [
        "you won't believe",
        "this will shock you",
        "what happened next",
        "doctors hate",
        "one weird trick",
        "secret that",
      ],
      // Conspiracy language
      conspiracy: [
        "cover-up",
        "exposed",
        "leaked",
        "hidden truth",
        "they don't want you to know",
        "mainstream media",
        "wake up",
      ],
      // Emotional manipulation
      emotional: ["outraged", "furious", "devastated", "heartbreaking", "terrifying", "disgusting"],
      // Urgency/scarcity
      urgency: ["breaking", "urgent", "immediate", "act now", "limited time", "before it's too late"],
      // Vague sources
      vague_sources: ["unnamed sources", "insider reveals", "expert says", "according to reports", "sources claim"],
    }

    const realIndicators = {
      // Credible sources
      credible: ["according to", "study shows", "research indicates", "published in", "peer-reviewed", "university"],
      // Official language
      official: ["announced", "confirmed", "reported", "stated", "official", "government", "department"],
      // Specific details
      specific: ["percent", "million", "billion", "study", "survey", "data", "statistics", "analysis"],
      // Professional tone
      professional: ["however", "furthermore", "additionally", "meanwhile", "consequently", "therefore"],
      // Factual reporting
      factual: ["investigation", "evidence", "findings", "results", "conclusion", "methodology"],
    }

    // Calculate scores with weights
    let fakeScore = 0
    let realScore = 0

    // Check fake indicators
    Object.entries(fakeIndicators).forEach(([category, words]) => {
      const weight = getWeight(category)
      words.forEach((word) => {
        if (processedText.includes(word)) {
          fakeScore += weight
        }
      })
    })

    // Check real indicators
    Object.entries(realIndicators).forEach(([category, words]) => {
      const weight = getWeight(category)
      words.forEach((word) => {
        if (processedText.includes(word)) {
          realScore += weight
        }
      })
    })

    // Additional analysis factors
    const textLength = processedText.length
    const wordCount = processedText.split(" ").length
    const avgWordLength = processedText.replace(/\s/g, "").length / wordCount

    // Structural analysis
    const hasQuestionMarks = (message.match(/\?/g) || []).length
    const hasExclamationMarks = (message.match(/!/g) || []).length
    const hasAllCaps = /[A-Z]{3,}/.test(message)
    const hasNumbers = /\d/.test(message)

    // Sentiment analysis (simplified)
    const positiveWords = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
    const negativeWords = ["bad", "terrible", "awful", "horrible", "disgusting", "outrageous"]

    let sentimentScore = 0
    positiveWords.forEach((word) => {
      if (processedText.includes(word)) sentimentScore += 1
    })
    negativeWords.forEach((word) => {
      if (processedText.includes(word)) sentimentScore -= 1
    })

    // Apply additional scoring rules
    if (textLength < 100) fakeScore += 2 // Very short articles are suspicious
    if (hasAllCaps) fakeScore += 3 // All caps indicates sensationalism
    if (hasExclamationMarks > 2) fakeScore += 2 // Too many exclamation marks
    if (hasQuestionMarks > 3) fakeScore += 1 // Too many questions
    if (!hasNumbers && textLength > 200) realScore += 1 // Real news often has statistics
    if (avgWordLength > 6) realScore += 1 // Longer words indicate more sophisticated writing

    // Calculate final prediction
    const totalScore = fakeScore - realScore
    const isFake = totalScore > 0

    // Calculate confidence based on score difference
    const scoreDifference = Math.abs(totalScore)
    let confidence = Math.min(0.95, 0.6 + scoreDifference * 0.05)

    // Adjust confidence based on text length and quality
    if (textLength < 50) confidence = Math.max(0.7, confidence) // Short texts are easier to classify
    if (textLength > 500) confidence = Math.min(0.9, confidence) // Longer texts are more complex

    return NextResponse.json({
      prediction: isFake ? "FAKE" : "REAL",
      confidence: confidence,
      processed_text: processedText.substring(0, 200),
      analysis: {
        fake_score: fakeScore,
        real_score: realScore,
        total_score: totalScore,
        text_length: textLength,
        word_count: wordCount,
        sentiment_score: sentimentScore,
        structural_flags: {
          has_all_caps: hasAllCaps,
          excessive_punctuation: hasExclamationMarks > 2 || hasQuestionMarks > 3,
          has_numbers: hasNumbers,
        },
      },
    })
  } catch (error) {
    console.error("Prediction error:", error)
    return NextResponse.json({ error: "Prediction failed" }, { status: 500 })
  }
}

function getWeight(category: string): number {
  const weights: { [key: string]: number } = {
    sensational: 3,
    clickbait: 4,
    conspiracy: 5,
    emotional: 2,
    urgency: 2,
    vague_sources: 3,
    credible: 4,
    official: 3,
    specific: 2,
    professional: 2,
    factual: 3,
  }
  return weights[category] || 1
}
