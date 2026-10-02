// lib/screens/history_detail_screen.dart
// --------------------------------------
// Detailed view of an archived analysis result from the database.

import 'package:flutter/material.dart';
import '../models/analysis_result.dart';
import 'result_screen.dart';

class HistoryDetailScreen extends StatelessWidget {
  final AnalysisResult result;

  const HistoryDetailScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    // Re-use the comprehensive ResultScreen layout
    return ResultScreen(result: result);
  }
}
