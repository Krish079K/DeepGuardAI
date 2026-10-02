import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../models/analysis_result.dart';

class RecentAnalysisCard extends StatelessWidget {
  final AnalysisResult result;
  final VoidCallback onTap;

  const RecentAnalysisCard({
    super.key,
    required this.result,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final isFake = result.isFake;
    final badgeColor = isFake ? const Color(0xFFEF4444) : const Color(0xFF10B981);
    final badgeBg = isFake
        ? const Color(0xFFEF4444).withOpacity(0.12)
        : const Color(0xFF10B981).withOpacity(0.12);

    final fakePct = (result.aiGeneratedProbability * 100).toStringAsFixed(1);
    final realPct = (result.realProbability * 100).toStringAsFixed(1);

    String formattedDate = '';
    if (result.createdAt != null) {
      formattedDate = DateFormat('MMM d, h:mm a').format(result.createdAt!.toLocal());
    }

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        splashColor: const Color(0xFF6C63FF).withOpacity(0.1),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFF111827),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: isFake
                  ? const Color(0xFFEF4444).withOpacity(0.2)
                  : const Color(0xFF1E293B),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: badgeBg,
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      isFake ? Icons.warning_amber_rounded : Icons.verified_user_rounded,
                      color: badgeColor,
                      size: 22,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          result.filename,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: Colors.white,
                          ),
                        ),
                        if (formattedDate.isNotEmpty)
                          Text(
                            formattedDate,
                            style: TextStyle(
                              fontSize: 11,
                              color: Colors.white.withOpacity(0.4),
                            ),
                          ),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: badgeBg,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: badgeColor.withOpacity(0.3)),
                    ),
                    child: Text(
                      isFake ? 'FAKE' : 'REAL',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        color: badgeColor,
                        letterSpacing: 0.5,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Real: $realPct%',
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF10B981),
                    ),
                  ),
                  Text(
                    'Fake: $fakePct%',
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFFEF4444),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: result.aiGeneratedProbability.clamp(0.0, 1.0),
                  backgroundColor: const Color(0xFF10B981).withOpacity(0.3),
                  valueColor: const AlwaysStoppedAnimation<Color>(Color(0xFFEF4444)),
                  minHeight: 6,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
