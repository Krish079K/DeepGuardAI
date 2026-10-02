// lib/models/analysis_result.dart
// --------------------------------
// Data model for a deepfake analysis result.

class AnalysisResult {
  final int? id;
  final String filename;
  final double visualScore;
  final double audioScore;
  final double lipSyncScore;
  final double temporalScore;
  final double realProbability;
  final double aiGeneratedProbability;
  final String prediction;
  final DateTime? createdAt;
  final double? processingTime;

  const AnalysisResult({
    this.id,
    required this.filename,
    required this.visualScore,
    required this.audioScore,
    required this.lipSyncScore,
    required this.temporalScore,
    required this.realProbability,
    required this.aiGeneratedProbability,
    required this.prediction,
    this.createdAt,
    this.processingTime,
  });

  bool get isFake => prediction.contains('AI-GENERATED');

  double get confidencePercent =>
      (isFake ? aiGeneratedProbability : realProbability) * 100;

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    return AnalysisResult(
      id:                      json['id'] as int?,
      filename:                json['filename'] as String? ?? 'unknown',
      visualScore:             _toDouble(json['visual_score']),
      audioScore:              _toDouble(json['audio_score']),
      lipSyncScore:            _toDouble(json['lip_sync_score']),
      temporalScore:           _toDouble(json['temporal_score']),
      realProbability:         _toDouble(json['real_probability']),
      aiGeneratedProbability:  _toDouble(json['ai_generated_probability']),
      prediction:              json['prediction'] as String? ?? 'UNKNOWN',
      createdAt:               json['created_at'] != null
                                 ? DateTime.tryParse(json['created_at'] as String)
                                 : null,
      processingTime:          json['processing_time_seconds'] != null
                                 ? _toDouble(json['processing_time_seconds'])
                                 : null,
    );
  }

  Map<String, dynamic> toJson() => {
    'id':                       id,
    'filename':                 filename,
    'visual_score':             visualScore,
    'audio_score':              audioScore,
    'lip_sync_score':           lipSyncScore,
    'temporal_score':           temporalScore,
    'real_probability':         realProbability,
    'ai_generated_probability': aiGeneratedProbability,
    'prediction':               prediction,
    'created_at':               createdAt?.toIso8601String(),
  };

  static double _toDouble(dynamic v) {
    if (v == null) return 0.5;
    if (v is double) return v;
    if (v is int) return v.toDouble();
    return double.tryParse(v.toString()) ?? 0.5;
  }
}
