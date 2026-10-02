// lib/screens/result_screen.dart
// ------------------------------
// Comprehensive result screen showing multimodal AI deepfake analysis.

import 'package:flutter/material.dart';
import '../models/analysis_result.dart';

class ResultScreen extends StatelessWidget {
  final AnalysisResult result;

  const ResultScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final isFake = result.isFake;
    final primaryColor = isFake ? const Color(0xFFEF4444) : const Color(0xFF10B981);
    final accentBg = primaryColor.withOpacity(0.12);

    final fakePct = (result.aiGeneratedProbability * 100).toStringAsFixed(1);
    final realPct = (result.realProbability * 100).toStringAsFixed(1);

    return Scaffold(
      backgroundColor: const Color(0xFF090E1A),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded, color: Colors.white, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Analysis Report',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: Colors.white),
        ),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Top Verdict Hero Card
            Container(
              padding: const EdgeInsets.symmetric(vertical: 28, horizontal: 20),
              decoration: BoxDecoration(
                color: const Color(0xFF111827),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: primaryColor.withOpacity(0.35), width: 1.5),
                boxShadow: [
                  BoxShadow(
                    color: primaryColor.withOpacity(0.15),
                    blurRadius: 24,
                    spreadRadius: 2,
                  ),
                ],
              ),
              child: Column(
                children: [
                  Container(
                    width: 76,
                    height: 76,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: accentBg,
                      border: Border.all(color: primaryColor.withOpacity(0.5), width: 2),
                    ),
                    child: Icon(
                      isFake ? Icons.warning_rounded : Icons.verified_rounded,
                      color: primaryColor,
                      size: 42,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    isFake ? 'AI-GENERATED / DEEPFAKE' : 'AUTHENTIC / REAL',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.w900,
                      color: primaryColor,
                      letterSpacing: 1.2,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    result.filename,
                    textAlign: TextAlign.center,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(fontSize: 13, color: Colors.white.withOpacity(0.6)),
                  ),
                  if (result.processingTime != null) ...[
                    const SizedBox(height: 4),
                    Text(
                      'Processed in ${result.processingTime!.toStringAsFixed(2)}s',
                      style: TextStyle(fontSize: 11, color: Colors.white.withOpacity(0.4)),
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Probability Distribution Card
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFF111827),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: const Color(0xFF1E293B)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Overall Probability',
                    style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Colors.white),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _buildProbabilityItem('Real Probability', '$realPct%', const Color(0xFF10B981)),
                      _buildProbabilityItem('AI Generated', '$fakePct%', const Color(0xFFEF4444)),
                    ],
                  ),
                  const SizedBox(height: 14),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: SizedBox(
                      height: 10,
                      child: LinearProgressIndicator(
                        value: result.aiGeneratedProbability.clamp(0.0, 1.0),
                        backgroundColor: const Color(0xFF10B981),
                        valueColor: const AlwaysStoppedAnimation<Color>(Color(0xFFEF4444)),
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('0% Fake (Authentic)',
                          style: TextStyle(fontSize: 10, color: Colors.white.withOpacity(0.4))),
                      Text('100% Fake (Manipulated)',
                          style: TextStyle(fontSize: 10, color: Colors.white.withOpacity(0.4))),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // 4 Multimodal Branches Breakdown
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFF111827),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: const Color(0xFF1E293B)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.analytics_outlined, color: Color(0xFF6C63FF), size: 20),
                      const SizedBox(width: 8),
                      const Text(
                        'Multimodal Branch Scores',
                        style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Colors.white),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Individual neural network branch predictions (Fake Probability)',
                    style: TextStyle(fontSize: 11, color: Colors.white.withOpacity(0.4)),
                  ),
                  const SizedBox(height: 20),
                  _buildBranchRow(
                    icon: Icons.remove_red_eye_rounded,
                    title: 'Visual Analysis',
                    subtitle: 'Facial artifact & compression detection',
                    score: result.visualScore,
                    color: const Color(0xFF6C63FF),
                  ),
                  const Divider(color: Color(0xFF1E293B), height: 28),
                  _buildBranchRow(
                    icon: Icons.mic_rounded,
                    title: 'Audio Analysis',
                    subtitle: 'Voice synthesis & spectral artifact model',
                    score: result.audioScore,
                    color: const Color(0xFF38BDF8),
                  ),
                  const Divider(color: Color(0xFF1E293B), height: 28),
                  _buildBranchRow(
                    icon: Icons.face_retouching_natural_rounded,
                    title: 'Lip-Sync Analysis',
                    subtitle: 'Phoneme-viseme temporal synchrony',
                    score: result.lipSyncScore,
                    color: const Color(0xFFF59E0B),
                  ),
                  const Divider(color: Color(0xFF1E293B), height: 28),
                  _buildBranchRow(
                    icon: Icons.timeline_rounded,
                    title: 'Temporal Analysis',
                    subtitle: 'Inter-frame consistency & jitter LSTM',
                    score: result.temporalScore,
                    color: const Color(0xFFA855F7),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Fusion Model Explanation Card
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFF111827),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: const Color(0xFF1E293B)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.hub_outlined, color: Color(0xFF10B981), size: 20),
                      const SizedBox(width: 8),
                      const Text(
                        'Multimodal Fusion Decision',
                        style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Colors.white),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Text(
                    isFake
                        ? 'The multimodal neural fusion network detected significant anomalies across visual, audio, or temporal dimensions indicative of AI-generated content.'
                        : 'The multimodal neural fusion network verified natural consistency across visual frames, acoustic spectrograms, and temporal coherence.',
                    style: TextStyle(fontSize: 13, height: 1.5, color: Colors.white.withOpacity(0.7)),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 28),

            // Action Button
            ElevatedButton(
              onPressed: () => Navigator.pop(context),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF6C63FF),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                elevation: 4,
              ),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.arrow_back_rounded, size: 20),
                  SizedBox(width: 8),
                  Text(
                    'Analyze Another Video',
                    style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildProbabilityItem(String label, String value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(fontSize: 12, color: Colors.white.withOpacity(0.5)),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(fontSize: 24, fontWeight: FontWeight.w900, color: color),
        ),
      ],
    );
  }

  Widget _buildBranchRow({
    required IconData icon,
    required String title,
    required String subtitle,
    required double score,
    required Color color,
  }) {
    final pct = (score * 100).toStringAsFixed(1);
    final isBranchFake = score >= 0.5;

    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: color.withOpacity(0.12),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, color: color, size: 22),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.white),
              ),
              const SizedBox(height: 2),
              Text(
                subtitle,
                style: TextStyle(fontSize: 11, color: Colors.white.withOpacity(0.4)),
              ),
              const SizedBox(height: 6),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: score.clamp(0.0, 1.0),
                  backgroundColor: const Color(0xFF1E293B),
                  valueColor: AlwaysStoppedAnimation<Color>(
                    isBranchFake ? const Color(0xFFEF4444) : const Color(0xFF10B981),
                  ),
                  minHeight: 5,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 14),
        Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              '$pct%',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w800,
                color: isBranchFake ? const Color(0xFFEF4444) : const Color(0xFF10B981),
              ),
            ),
            Text(
              isBranchFake ? 'Suspicious' : 'Authentic',
              style: TextStyle(
                fontSize: 10,
                color: (isBranchFake ? const Color(0xFFEF4444) : const Color(0xFF10B981)).withOpacity(0.8),
              ),
            ),
          ],
        ),
      ],
    );
  }
}
