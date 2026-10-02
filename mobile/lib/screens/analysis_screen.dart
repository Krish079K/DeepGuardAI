// lib/screens/analysis_screen.dart
// ----------------------------------
// Upload screen shown while the video is being analysed.
// Shows animated step-by-step progress messages.

import 'dart:io';

import 'package:flutter/material.dart';

import '../models/analysis_result.dart';
import '../services/api_service.dart';
import 'result_screen.dart';

class AnalysisScreen extends StatefulWidget {
  final File videoFile;
  const AnalysisScreen({super.key, required this.videoFile});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen>
    with SingleTickerProviderStateMixin {
  final _api = ApiService();

  // Analysis steps shown to the user
  static const _steps = [
    (Icons.upload_rounded,          'Preparing video…'),
    (Icons.remove_red_eye_rounded,  'Analyzing visual features…'),
    (Icons.mic_rounded,             'Analyzing audio…'),
    (Icons.face_rounded,            'Checking lip synchronization…'),
    (Icons.timeline_rounded,        'Analyzing temporal features…'),
    (Icons.merge_type_rounded,      'Running multimodal fusion…'),
    (Icons.check_circle_rounded,    'Analysis complete.'),
  ];

  int    _currentStep = 0;
  bool   _uploading   = true;
  String _errorMsg    = '';

  late AnimationController _spinCtrl;
  late Animation<double>   _spinAnim;

  @override
  void initState() {
    super.initState();
    _spinCtrl = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();
    _spinAnim = Tween<double>(begin: 0.0, end: 1.0).animate(_spinCtrl);

    _startAnalysis();
  }

  @override
  void dispose() {
    _spinCtrl.dispose();
    super.dispose();
  }

  Future<void> _startAnalysis() async {
    // Simulate step progression while upload is happening
    _advanceSteps();

    try {
      final result = await _api.analyzeVideo(widget.videoFile);
      if (!mounted) return;
      setState(() {
        _currentStep = _steps.length - 1;
        _uploading   = false;
      });
      await Future.delayed(const Duration(milliseconds: 600));
      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        PageRouteBuilder(
          pageBuilder: (_, __, ___) => ResultScreen(result: result),
          transitionsBuilder: (_, anim, __, child) =>
              SlideTransition(
                position: Tween<Offset>(
                  begin: const Offset(1, 0),
                  end: Offset.zero,
                ).animate(CurvedAnimation(
                    parent: anim, curve: Curves.easeOutCubic)),
                child: child,
              ),
          transitionDuration: const Duration(milliseconds: 500),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _uploading = false;
        _errorMsg  = e.toString().replaceFirst('ApiException(500): ', '');
      });
    }
  }

  // Advance through step labels on a timer (cosmetic)
  Future<void> _advanceSteps() async {
    // Steps 0–5 are shown during upload (step 6 = complete, shown in code above)
    for (int i = 0; i <= 5; i++) {
      await Future.delayed(Duration(
        milliseconds: i == 0 ? 300 : (i == 5 ? 100 : 2000),
      ));
      if (!mounted) break;
      setState(() => _currentStep = i);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF090E1A),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 28),
          child: _errorMsg.isNotEmpty ? _buildError() : _buildProgress(),
        ),
      ),
    );
  }

  Widget _buildProgress() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        const SizedBox(height: 60),

        // Animated shield
        RotationTransition(
          turns: _spinAnim,
          child: Container(
            width: 90, height: 90,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: const SweepGradient(
                colors: [Color(0xFF6C63FF), Color(0xFF3ECFCF),
                         Color(0xFF6C63FF)],
              ),
              boxShadow: [
                BoxShadow(
                  color: const Color(0xFF6C63FF).withOpacity(0.4),
                  blurRadius: 30,
                  spreadRadius: 4,
                ),
              ],
            ),
            child: const Icon(Icons.shield_rounded,
                color: Colors.white, size: 42),
          ),
        ),
        const SizedBox(height: 32),

        const Text(
          'ANALYZING VIDEO',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w900,
            color: Colors.white,
            letterSpacing: 3,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          widget.videoFile.path.split(Platform.pathSeparator).last,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(
            fontSize: 12, color: Color(0xFF8A9BB0),
          ),
        ),

        const SizedBox(height: 48),

        // Step list
        Expanded(
          child: ListView.builder(
            itemCount: _steps.length,
            itemBuilder: (ctx, i) {
              final done    = i < _currentStep;
              final active  = i == _currentStep;
              final pending = i > _currentStep;

              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: AnimatedOpacity(
                  opacity: pending ? 0.35 : 1.0,
                  duration: const Duration(milliseconds: 300),
                  child: Row(
                    children: [
                      // Step icon
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 300),
                        width: 40, height: 40,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: done
                              ? const Color(0xFF00E676).withOpacity(0.15)
                              : active
                                  ? const Color(0xFF6C63FF).withOpacity(0.2)
                                  : const Color(0xFF1E293B),
                          border: Border.all(
                            color: done
                                ? const Color(0xFF00E676)
                                : active
                                    ? const Color(0xFF6C63FF)
                                    : const Color(0xFF2D3748),
                            width: 1.5,
                          ),
                        ),
                        child: Icon(
                          done
                              ? Icons.check_rounded
                              : _steps[i].$1,
                          size: 18,
                          color: done
                              ? const Color(0xFF00E676)
                              : active
                                  ? const Color(0xFF6C63FF)
                                  : const Color(0xFF4A5568),
                        ),
                      ),
                      const SizedBox(width: 16),
                      // Label
                      Expanded(
                        child: Text(
                          _steps[i].$2,
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: active
                                ? FontWeight.w600
                                : FontWeight.w400,
                            color: done
                                ? const Color(0xFF00E676)
                                : active
                                    ? Colors.white
                                    : const Color(0xFF4A5568),
                          ),
                        ),
                      ),
                      if (active)
                        const SizedBox(
                          width: 16, height: 16,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Color(0xFF6C63FF),
                          ),
                        ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),

        Padding(
          padding: const EdgeInsets.only(bottom: 24),
          child: Text(
            'This may take 30–120 seconds depending on video length',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 11,
              color: Colors.white.withOpacity(0.3),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildError() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(Icons.error_outline_rounded,
            size: 64, color: Color(0xFFFF5252)),
        const SizedBox(height: 20),
        const Text('Analysis Failed',
            style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w700,
                color: Colors.white)),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFF1A1A2E),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: const Color(0xFFFF5252).withOpacity(0.3)),
          ),
          child: Text(
            _errorMsg,
            textAlign: TextAlign.center,
            style: const TextStyle(
                fontSize: 13, color: Color(0xFF8A9BB0), height: 1.5),
          ),
        ),
        const SizedBox(height: 24),
        Row(
          children: [
            Expanded(
              child: OutlinedButton(
                onPressed: () => Navigator.pop(context),
                style: OutlinedButton.styleFrom(
                  foregroundColor: const Color(0xFF8A9BB0),
                  side: const BorderSide(color: Color(0xFF2D3748)),
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text('Go Back'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: ElevatedButton(
                onPressed: () {
                  setState(() {
                    _currentStep = 0;
                    _uploading   = true;
                    _errorMsg    = '';
                  });
                  _startAnalysis();
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF6C63FF),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text('Retry'),
              ),
            ),
          ],
        ),
      ],
    );
  }
}
