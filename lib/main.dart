import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:url_launcher/url_launcher.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AnGiangLensApp());
}

class RecognitionResult {
  final String label;
  final double similarity;
  const RecognitionResult(this.label, this.similarity);
}

class PlaceInfo {
  final String id;
  final String name;
  final String location;
  final String icon;
  final String description;
  final String videoUrl;
  final List<String> images;

  const PlaceInfo({
    required this.id,
    required this.name,
    required this.location,
    required this.icon,
    required this.description,
    required this.videoUrl,
    required this.images,
  });

  factory PlaceInfo.fromJson(String id, Map<String, dynamic> json) {
    return PlaceInfo(
      id: id,
      name: json['name']?.toString() ?? id,
      location: json['location']?.toString() ?? '',
      icon: json['icon']?.toString() ?? '📍',
      description: json['description']?.toString() ?? '',
      videoUrl: json['videoUrl']?.toString() ?? '',
      images: (json['images'] as List<dynamic>? ?? const [])
          .map((e) => e.toString())
          .toList(),
    );
  }
}

class _Sample {
  final String label;
  final List<double> embedding;
  _Sample(this.label, this.embedding);
}

class _ScoredSample {
  final String label;
  final double score;
  _ScoredSample(this.label, this.score);
}

class AiService {
  Interpreter? _interpreter;

  final List<_Sample> _samples = [];

  double _threshold = 0.75;

  double get threshold => _threshold;

  bool get isReady =>
      _interpreter != null && _samples.isNotEmpty;

  // =====================================================
  // KHỞI TẠO MODEL + DATABASE
  // =====================================================

  Future<void> initialize() async {
    _interpreter = await Interpreter.fromAsset(
      'assets/ai/angiang_lens_model.tflite',
    );

    final raw = await rootBundle.loadString(
      'assets/ai/angiang_lens_database.json',
    );

    final data =
        jsonDecode(raw) as Map<String, dynamic>;

    _threshold =
        (data['threshold'] as num?)?.toDouble() ??
            0.75;

    final items =
        data['items'] as List<dynamic>? ??
            const [];

    _samples
      ..clear()
      ..addAll(
        items.map(
          (item) {
            final m =
                item as Map<String, dynamic>;

            return _Sample(
              m['label'].toString(),
              (m['embedding'] as List<dynamic>)
                  .map(
                    (v) =>
                        (v as num).toDouble(),
                  )
                  .toList(
                    growable: false,
                  ),
            );
          },
        ),
      );

    if (_samples.isEmpty) {
      throw StateError(
        'Database AI không có ảnh mẫu.',
      );
    }

    final outputDim =
        _interpreter!
            .getOutputTensor(0)
            .shape
            .last;

    if (_samples.first.embedding.length !=
        outputDim) {
      throw StateError(
        'Embedding database '
        '(${_samples.first.embedding.length}) '
        'không khớp output model '
        '($outputDim).',
      );
    }
  }

  // =====================================================
  // NHẬN DIỆN
  // =====================================================

  Future<RecognitionResult?> recognize(
    Uint8List bytes,
  ) async {
    if (!isReady) {
      throw StateError(
        'AI chưa sẵn sàng.',
      );
    }

    img.Image? decoded =
        img.decodeImage(bytes);

    if (decoded == null) {
      throw StateError(
        'Không đọc được ảnh.',
      );
    }

    // Sửa chiều ảnh camera theo EXIF
    decoded =
        img.bakeOrientation(decoded);

    // ===================================================
    // TẠO NHIỀU PHIÊN BẢN ẢNH
    // ===================================================

    final variants = <img.Image>[
      // 1. Toàn bộ ảnh
      decoded,

      // 2. Crop nhẹ
      _centerCrop(
        decoded,
        0.92,
      ),

      // 3. Crop vừa
      _centerCrop(
        decoded,
        0.84,
      ),

      // 4. Crop mạnh
      _centerCrop(
        decoded,
        0.76,
      ),

      // 5. Loại bớt thanh menu/viền
      // khi chụp màn hình máy tính
      _screenCrop(
        decoded,
      ),
    ];

    RecognitionResult? bestResult;

    // ===================================================
    // CHẠY AI CHO TỪNG PHIÊN BẢN
    // ===================================================

    for (var i = 0;
        i < variants.length;
        i++) {
      final query =
          _getEmbedding(
        variants[i],
      );

      final result =
          _compareEmbedding(
        query,
      );

      print(
        'Variant $i: '
        '${result.label} '
        '${(result.similarity * 100).toStringAsFixed(1)}%',
      );

      if (bestResult == null ||
          result.similarity >
              bestResult.similarity) {
        bestResult = result;
      }
    }

    if (bestResult == null) {
      return null;
    }

    print(
      'BEST = '
      '${bestResult.label} '
      '${(bestResult.similarity * 100).toStringAsFixed(1)}%',
    );

    // ===================================================
    // GIỮ NGƯỠNG 75%
    // ===================================================

    if (bestResult.similarity <
        _threshold) {
      return null;
    }

    return bestResult;
  }

  // =====================================================
  // CROP GIỮA ẢNH
  // =====================================================

  img.Image _centerCrop(
    img.Image source,
    double ratio,
  ) {
    final cropWidth =
        (source.width * ratio).round();

    final cropHeight =
        (source.height * ratio).round();

    final x =
        ((source.width - cropWidth) / 2)
            .round();

    final y =
        ((source.height - cropHeight) / 2)
            .round();

    return img.copyCrop(
      source,
      x: x,
      y: y,
      width: cropWidth,
      height: cropHeight,
    );
  }

  // =====================================================
  // CROP DÀNH CHO ẢNH CHỤP MÀN HÌNH
  // =====================================================

  img.Image _screenCrop(
    img.Image source,
  ) {
    // Bỏ khoảng 4% hai bên
    final left =
        (source.width * 0.04).round();

    final right =
        (source.width * 0.04).round();

    // Bỏ nhiều hơn ở phía trên
    // vì thường có thanh menu máy tính
    final top =
        (source.height * 0.10).round();

    final bottom =
        (source.height * 0.04).round();

    final cropWidth =
        source.width -
        left -
        right;

    final cropHeight =
        source.height -
        top -
        bottom;

    return img.copyCrop(
      source,
      x: left,
      y: top,
      width: cropWidth,
      height: cropHeight,
    );
  }

  // =====================================================
  // TẠO EMBEDDING
  // =====================================================

  List<double> _getEmbedding(
    img.Image source,
  ) {
    final inputShape =
        _interpreter!
            .getInputTensor(0)
            .shape;

    if (inputShape.length != 4 ||
        inputShape[0] != 1 ||
        inputShape[3] != 3) {
      throw StateError(
        'Input shape không hỗ trợ: '
        '$inputShape',
      );
    }

    final h =
        inputShape[1];

    final w =
        inputShape[2];

    // Resize giống bản Python/Gradio
    final resized =
        img.copyResize(
      source,
      width: w,
      height: h,
      interpolation:
          img.Interpolation.linear,
    );

    final flat =
        Float32List(
      w * h * 3,
    );

    var offset = 0;

    for (var y = 0;
        y < h;
        y++) {
      for (var x = 0;
          x < w;
          x++) {
        final p =
            resized.getPixel(
          x,
          y,
        );

        flat[offset++] =
            p.r.toDouble();

        flat[offset++] =
            p.g.toDouble();

        flat[offset++] =
            p.b.toDouble();
      }
    }

    final input =
        flat.reshape(
      [
        1,
        h,
        w,
        3,
      ],
    );

    final outDim =
        _interpreter!
            .getOutputTensor(0)
            .shape
            .last;

    final output =
        Float32List(outDim)
            .reshape(
      [
        1,
        outDim,
      ],
    );

    _interpreter!.run(
      input,
      output,
    );

    final rawEmbedding =
        (output[0] as List)
            .map(
              (v) =>
                  (v as num).toDouble(),
            )
            .toList(
              growable: false,
            );

    return _normalize(
      rawEmbedding,
    );
  }

  // =====================================================
  // SO SÁNH DATABASE - TOP 5
  // =====================================================

  RecognitionResult _compareEmbedding(
    List<double> query,
  ) {
    final scored =
        _samples
            .map(
              (s) =>
                  _ScoredSample(
                s.label,
                _dot(
                  query,
                  s.embedding,
                ),
              ),
            )
            .toList()
          ..sort(
            (a, b) =>
                b.score.compareTo(
              a.score,
            ),
          );

    final top =
        scored
            .take(
              math.min(
                5,
                scored.length,
              ),
            )
            .toList();

    // ===================================================
    // ĐẾM PHIẾU TOP 5
    // ===================================================

    final counts =
        <String, int>{};

    for (final item in top) {
      counts[item.label] =
          (counts[item.label] ?? 0) +
              1;
    }

    var winner =
        top.first.label;

    var bestCount =
        counts[winner] ?? 0;

    for (final item in top) {
      final count =
          counts[item.label] ?? 0;

      if (count > bestCount) {
        winner =
            item.label;

        bestCount =
            count;
      }
    }

    // ===================================================
    // TÍNH ĐỘ TƯƠNG ĐỒNG TRUNG BÌNH
    // ===================================================

    final winnerScores =
        top
            .where(
              (e) =>
                  e.label == winner,
            )
            .map(
              (e) =>
                  e.score,
            )
            .toList();

    final similarity =
        winnerScores.reduce(
              (a, b) =>
                  a + b,
            ) /
            winnerScores.length;

    return RecognitionResult(
      winner,
      similarity,
    );
  }

  // =====================================================
  // NORMALIZE
  // =====================================================

  List<double> _normalize(
    List<double> v,
  ) {
    var sumSq = 0.0;

    for (final x in v) {
      sumSq += x * x;
    }

    final norm =
        math.sqrt(sumSq) +
            1e-10;

    return v
        .map(
          (x) =>
              x / norm,
        )
        .toList(
          growable: false,
        );
  }

  // =====================================================
  // COSINE SIMILARITY
  // =====================================================

  double _dot(
    List<double> a,
    List<double> b,
  ) {
    final n =
        math.min(
      a.length,
      b.length,
    );

    var s = 0.0;

    for (var i = 0;
        i < n;
        i++) {
      s +=
          a[i] * b[i];
    }

    return s
        .clamp(
          -1.0,
          1.0,
        )
        .toDouble();
  }

  // =====================================================
  // ĐÓNG MODEL
  // =====================================================

  void dispose() {
    _interpreter?.close();

    _interpreter = null;
  }
}

class PlaceRepository {
  final Map<String, PlaceInfo> _places = {};

  Future<void> initialize() async {
    final raw = await rootBundle.loadString('assets/content/place_catalog.json');
    final data = jsonDecode(raw) as Map<String, dynamic>;
    _places
      ..clear()
      ..addEntries(data.entries.map((e) {
        return MapEntry(
          e.key,
          PlaceInfo.fromJson(e.key, e.value as Map<String, dynamic>),
        );
      }));
  }

  PlaceInfo? byId(String id) => _places[id];
}

class AnGiangLensApp extends StatelessWidget {
  const AnGiangLensApp({super.key});

  @override
  Widget build(BuildContext context) {
    final scheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFF087F5B),
      brightness: Brightness.light,
    );

    return MaterialApp(
      title: 'AnGiang Lens',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: scheme,
        scaffoldBackgroundColor: const Color(0xFFF4F8F6),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            backgroundColor: const Color(0xFF087F5B),
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: const Color(0xFF087F5B),
            side: const BorderSide(color: Color(0xFF83B9A7)),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
          ),
        ),
      ),
      home: const ScannerPage(),
    );
  }
}

class ScannerPage extends StatefulWidget {
  const ScannerPage({super.key});

  @override
  State<ScannerPage> createState() => _ScannerPageState();
}

class _ScannerPageState extends State<ScannerPage> {
  final ImagePicker _picker = ImagePicker();
  final AiService _ai = AiService();
  final PlaceRepository _places = PlaceRepository();

  bool _loading = true;
  bool _recognizing = false;
  String? _error;
  XFile? _selected;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      await Future.wait([_ai.initialize(), _places.initialize()]);
    } catch (e) {
      _error = e.toString();
    }
    if (mounted) setState(() => _loading = false);
  }

  Future<void> _pickAndRecognize(ImageSource source) async {
    if (_loading || _recognizing || _error != null) return;

    try {
      final picked = await _picker.pickImage(
        source: source,
        imageQuality: 92,
        maxWidth: 1600,
      );
      if (picked == null) return;

      setState(() {
        _selected = picked;
        _recognizing = true;
      });

      final result = await _ai.recognize(await picked.readAsBytes());
      if (!mounted) return;

      if (result == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            backgroundColor: const Color(0xFF8B2F2F),
            content: Text(
              'Không nhận diện được địa điểm. Cần đạt ít nhất '
              '${(_ai.threshold * 100).toStringAsFixed(0)}% tương đồng.',
            ),
          ),
        );
        return;
      }

      final place = _places.byId(result.label);
      if (place == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Chưa có nội dung cho ${result.label}.')),
        );
        return;
      }

      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => PlaceDetailsPage(
            place: place,
            result: result,
            scannedImagePath: picked.path,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Lỗi nhận diện: $e')),
      );
    } finally {
      if (mounted) setState(() => _recognizing = false);
    }
  }

  @override
  void dispose() {
    _ai.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(14, 12, 14, 28),
          children: [
            Container(
              padding: const EdgeInsets.fromLTRB(22, 24, 22, 26),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    Color(0xFF075E4A),
                    Color(0xFF0B8A6A),
                    Color(0xFF18A58C),
                  ],
                ),
                borderRadius: BorderRadius.circular(28),
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x26075E4A),
                    blurRadius: 28,
                    offset: Offset(0, 14),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('🏞️', style: TextStyle(fontSize: 44)),
                  Text(
                    'AnGiang Lens',
                    style: theme.textTheme.headlineMedium?.copyWith(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                  const SizedBox(height: 6),
                  const Text(
                    'Khám phá An Giang qua trí tuệ nhân tạo',
                    style: TextStyle(
                      color: Color(0xFFE6FFF6),
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    'Chụp địa điểm bằng camera. AI chạy trực tiếp trên điện thoại và chỉ mở thông tin khi nhận diện thành công.',
                    style: TextStyle(color: Color(0xFFD1F5E8), height: 1.45),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 22),
            Text(
              '📷 Quét địa điểm',
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.w900,
                color: const Color(0xFF12372E),
              ),
            ),
            const SizedBox(height: 6),
            const Text(
              'Chụp rõ toàn cảnh, đủ ánh sáng và hạn chế vật cản.',
              style: TextStyle(color: Color(0xFF64748B)),
            ),
            const SizedBox(height: 14),
            Container(
              height: 330,
              decoration: BoxDecoration(
                color: const Color(0xFFE5F0EC),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: const Color(0xFFD5E4DE)),
              ),
              clipBehavior: Clip.antiAlias,
              child: _selected == null
                  ? const Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.camera_alt_outlined,
                            size: 70,
                            color: Color(0xFF4B7769),
                          ),
                          SizedBox(height: 14),
                          Text(
                            'Ảnh quét sẽ hiển thị tại đây',
                            style: TextStyle(
                              color: Color(0xFF4B7769),
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ],
                      ),
                    )
                  : Image.file(
                      File(_selected!.path),
                      fit: BoxFit.cover,
                      width: double.infinity,
                    ),
            ),
            const SizedBox(height: 16),
            if (_loading)
              const StateCard(
                icon: Icons.memory,
                title: 'Đang khởi tạo AI...',
                subtitle: 'Đang nạp model và kho đặc trưng.',
                progress: true,
              )
            else if (_error != null)
              StateCard(
                icon: Icons.error_outline,
                title: 'Không khởi tạo được AI',
                subtitle: _error!,
                error: true,
              )
            else if (_recognizing)
              const StateCard(
                icon: Icons.auto_awesome,
                title: 'AI đang phân tích...',
                subtitle: 'Đang so sánh Top-5 ảnh mẫu offline.',
                progress: true,
              )
            else
              const StateCard(
                icon: Icons.check_circle_outline,
                title: 'AI đã sẵn sàng',
                subtitle: 'Model và database đang chạy offline trên máy.',
              ),
            const SizedBox(height: 16),
            SizedBox(
              height: 56,
              child: FilledButton.icon(
                onPressed: _loading || _recognizing || _error != null
                    ? null
                    : () => _pickAndRecognize(ImageSource.camera),
                icon: const Icon(Icons.camera_alt),
                label: const Text(
                  'QUÉT BẰNG CAMERA',
                  style: TextStyle(fontWeight: FontWeight.w900),
                ),
              ),
            ),
            const SizedBox(height: 10),
            SizedBox(
              height: 52,
              child: OutlinedButton.icon(
                onPressed: _loading || _recognizing || _error != null
                    ? null
                    : () => _pickAndRecognize(ImageSource.gallery),
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text('Chọn ảnh có sẵn để thử'),
              ),
            ),
            const SizedBox(height: 18),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFFEAF6F1),
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: const Color(0xFFCDE8DC)),
              ),
              child: const Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.info_outline, color: Color(0xFF087F5B)),
                  SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'Ảnh không đủ giống Miếu Bà Chúa Xứ hoặc Hòn Phụ Tử sẽ bị từ chối, thay vì ép chọn sai địa điểm.',
                      style: TextStyle(color: Color(0xFF245B4B), height: 1.45),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class StateCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final bool progress;
  final bool error;

  const StateCard({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    this.progress = false,
    this.error = false,
  });

  @override
  Widget build(BuildContext context) {
    final fg = error ? const Color(0xFF9B2C2C) : const Color(0xFF176B52);
    final bg = error ? const Color(0xFFFFEEEE) : Colors.white;
    return Container(
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: error ? const Color(0xFFF3C2C2) : const Color(0xFFDDE9E4),
        ),
      ),
      child: Row(
        children: [
          Icon(icon, color: fg),
          const SizedBox(width: 11),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: TextStyle(color: fg, fontWeight: FontWeight.w800)),
                const SizedBox(height: 3),
                Text(subtitle, style: const TextStyle(color: Color(0xFF64748B), fontSize: 13)),
              ],
            ),
          ),
          if (progress) ...[
            const SizedBox(width: 10),
            const SizedBox(
              width: 22,
              height: 22,
              child: CircularProgressIndicator(strokeWidth: 2.5),
            ),
          ],
        ],
      ),
    );
  }
}

class PlaceDetailsPage extends StatefulWidget {
  final PlaceInfo place;
  final RecognitionResult result;
  final String scannedImagePath;

  const PlaceDetailsPage({
    super.key,
    required this.place,
    required this.result,
    required this.scannedImagePath,
  });

  @override
  State<PlaceDetailsPage> createState() => _PlaceDetailsPageState();
}

class _PlaceDetailsPageState extends State<PlaceDetailsPage> {
  final FlutterTts _tts = FlutterTts();
  bool _speaking = false;

  @override
  void initState() {
    super.initState();
    _prepareTts();
  }

  Future<void> _prepareTts() async {
    await _tts.setLanguage('vi-VN');
    await _tts.setSpeechRate(0.45);
    await _tts.setPitch(1.0);
    _tts.setCompletionHandler(() {
      if (mounted) setState(() => _speaking = false);
    });
    _tts.setCancelHandler(() {
      if (mounted) setState(() => _speaking = false);
    });
  }

  Future<void> _toggleSpeech() async {
    if (_speaking) {
      await _tts.stop();
      if (mounted) setState(() => _speaking = false);
      return;
    }
    if (mounted) setState(() => _speaking = true);
    await _tts.speak('${widget.place.name}. ${widget.place.description}');
  }

  Future<void> _openVideo() async {
    final value = widget.place.videoUrl.trim();
    if (value.isEmpty) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Chưa thêm video cho địa điểm này.')),
      );
      return;
    }
    final uri = Uri.tryParse(value);
    if (uri == null || !await launchUrl(uri)) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Không mở được video.')),
      );
    }
  }

  @override
  void dispose() {
    _tts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final pct = (widget.result.similarity * 100).clamp(0, 100);

    return Scaffold(
      appBar: AppBar(title: const Text('AnGiang Lens')),
      body: ListView(
        padding: EdgeInsets.zero,
        children: [
          Stack(
            alignment: Alignment.bottomLeft,
            children: [
              AspectRatio(
                aspectRatio: 16 / 10,
                child: Image.file(
                  File(widget.scannedImagePath),
                  fit: BoxFit.cover,
                ),
              ),
              Container(
                height: 150,
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [Colors.transparent, Color(0xCC063D32)],
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(20),
                child: Text(
                  '${widget.place.icon}  ${widget.place.name}',
                  style: theme.textTheme.headlineSmall?.copyWith(
                    color: Colors.white,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ],
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 18, 18, 28),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: [
                    pill(Icons.location_on_outlined, widget.place.location),
                    pill(Icons.auto_awesome, 'AI tương đồng ${pct.toStringAsFixed(1)}%'),
                  ],
                ),
                const SizedBox(height: 22),
                cardSection(
                  '📖 Giới thiệu',
                  Text(
                    widget.place.description,
                    style: theme.textTheme.bodyLarge?.copyWith(
                      height: 1.65,
                      color: const Color(0xFF334155),
                    ),
                  ),
                ),
                const SizedBox(height: 14),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    onPressed: _toggleSpeech,
                    icon: Icon(_speaking ? Icons.stop_circle : Icons.volume_up),
                    label: Text(_speaking ? 'Dừng thuyết minh' : 'Nghe thuyết minh'),
                  ),
                ),
                const SizedBox(height: 22),
                Text(
                  '🖼 Hình ảnh địa điểm',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w800,
                    color: const Color(0xFF12372E),
                  ),
                ),
                const SizedBox(height: 10),
                if (widget.place.images.isEmpty)
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: const Color(0xFFDCE7E2)),
                    ),
                    child: const Text(
                      'Bản demo dùng ảnh vừa quét làm ảnh đại diện. Khi bạn gửi thêm ảnh địa điểm, có thể đưa chúng vào thư viện offline.',
                    ),
                  )
                else
                  SizedBox(
                    height: 140,
                    child: ListView.separated(
                      scrollDirection: Axis.horizontal,
                      itemCount: widget.place.images.length,
                      separatorBuilder: (_, __) => const SizedBox(width: 10),
                      itemBuilder: (context, index) => ClipRRect(
                        borderRadius: BorderRadius.circular(16),
                        child: Image.asset(
                          widget.place.images[index],
                          width: 190,
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
                  ),
                const SizedBox(height: 22),
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    onPressed: _openVideo,
                    icon: const Icon(Icons.play_circle_outline),
                    label: const Text('Xem video giới thiệu'),
                  ),
                ),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton.tonalIcon(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.camera_alt_outlined),
                    label: const Text('Quét địa điểm khác'),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget pill(IconData icon, String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
      decoration: BoxDecoration(
        color: const Color(0xFFE5F5EF),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 17, color: const Color(0xFF087F5B)),
          const SizedBox(width: 6),
          Text(
            text,
            style: const TextStyle(
              color: Color(0xFF087F5B),
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }

  Widget cardSection(String title, Widget child) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: const Color(0xFFDDE9E4)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w800,
              color: Color(0xFF12372E),
            ),
          ),
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }
}
