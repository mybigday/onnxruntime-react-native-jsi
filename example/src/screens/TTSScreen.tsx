import { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
} from 'react-native';
import { useAI } from '../context/AIContext';
import { encodeWavPCM16Mono } from '../utils/wav';
import { AudioContext } from 'react-native-audio-api';

export default function TTSScreen() {
  const { runTTS, isLoaded } = useAI();
  const [input, setInput] = useState('Hello from TTS!');
  const [result, setResult] = useState<string>('');
  const [isRunning, setIsRunning] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [buffer, setBuffer] = useState<any | undefined>(undefined);
  const scrollRef = useRef<ScrollView | null>(null);
  const audioCtxRef = useRef<any | null>(null);
  const sourceNodeRef = useRef<any | null>(null);
  const rafRef = useRef<number | null>(null);
  const [progressMs, setProgressMs] = useState(0);

  useEffect(() => {
    audioCtxRef.current = new AudioContext();
    return () => {
      audioCtxRef.current?.close();
    };
  }, []);

  const onRun = useCallback(async () => {
    if (!isLoaded) return;
    setIsRunning(true);
    setResult('');
    try {
      const out = await runTTS(input);
      if (audioCtxRef.current && out?.audio && out.sampling_rate) {
        const wav = encodeWavPCM16Mono(out.audio, out.sampling_rate);
        const audioBuffer = await audioCtxRef.current.decodeAudioData(wav);
        setBuffer(audioBuffer);
        setProgressMs(0);
        setResult(
          `Generated ${out.audio.length} samples @ ${out.sampling_rate} Hz`
        );
      } else {
        setResult('No audio returned');
      }
    } catch (e: any) {
      setResult(String(e?.message ?? e));
    } finally {
      setIsRunning(false);
      requestAnimationFrame(() =>
        scrollRef.current?.scrollToEnd({ animated: true })
      );
    }
  }, [isLoaded, input, runTTS]);

  const play = useCallback(() => {
    if (!audioCtxRef.current || !buffer) return;
    // Stop any previous playback
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.stop();
      } catch {}
      sourceNodeRef.current = null;
    }
    const src = audioCtxRef.current.createBufferSource();
    src.buffer = buffer;
    src.connect(audioCtxRef.current.destination);
    src.start();
    sourceNodeRef.current = src;
    setIsPlaying(true);
    setProgressMs(0);
    const startTime = audioCtxRef.current.currentTime;
    const durationMs = buffer.duration * 1000;
    const tick = () => {
      if (!audioCtxRef.current) return;
      // Ensure we only track the currently active source
      if (sourceNodeRef.current !== src) return;
      const elapsedMs = Math.min(
        (audioCtxRef.current.currentTime - startTime) * 1000,
        durationMs
      );
      setProgressMs(elapsedMs);
      if (elapsedMs >= durationMs) {
        // Fallback in case onended doesn't fire
        setIsPlaying(false);
        sourceNodeRef.current = null;
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    src.onended = () => {
      // Only handle end for the active source
      if (sourceNodeRef.current === src) {
        setIsPlaying(false);
        sourceNodeRef.current = null;
        setProgressMs(durationMs);
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [buffer]);

  const stop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    sourceNodeRef.current?.stop();
    setIsPlaying(false);
  }, []);

  return (
    <View style={styles.container}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>TTS</Text>
      </View>
      <TextInput
        style={styles.input}
        value={input}
        onChangeText={setInput}
        placeholder={'Enter text to synthesize'}
        autoCapitalize="none"
        autoCorrect={false}
      />
      <TouchableOpacity
        style={styles.runBtn}
        onPress={onRun}
        disabled={!isLoaded || isRunning}
      >
        <Text style={styles.runBtnText}>
          {isRunning ? 'Running...' : 'Synthesize'}
        </Text>
      </TouchableOpacity>
      {buffer && (
        <View style={{ marginTop: 12 }}>
          <Text selectable numberOfLines={1}>
            Ready: {Math.round(buffer.duration * 1000)} ms
          </Text>
          <View style={styles.progressTrack}>
            <View
              style={[
                styles.progressFill,
                {
                  width: `${Math.min(100, Math.max(0, (progressMs / (buffer.duration * 1000)) * 100))}%`,
                },
              ]}
            />
          </View>
          <Text style={styles.progressText}>
            {Math.round(progressMs)} / {Math.round(buffer.duration * 1000)} ms
          </Text>
          <View style={[styles.row, { marginTop: 8 }]}>
            {!isPlaying ? (
              <TouchableOpacity style={styles.smallBtn} onPress={play}>
                <Text style={styles.smallBtnText}>Play</Text>
              </TouchableOpacity>
            ) : (
              <TouchableOpacity
                style={[styles.smallBtn, { backgroundColor: '#B71C1C' }]}
                onPress={stop}
              >
                <Text style={styles.smallBtnText}>Stop</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>
      )}
      <ScrollView style={styles.output} ref={scrollRef}>
        <Text selectable>{result}</Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16 },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: { fontSize: 20 },
  memBtn: {
    backgroundColor: '#6D4C41',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
  },
  input: {
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginTop: 12,
  },
  row: { flexDirection: 'row', alignItems: 'center', marginTop: 8 },
  smallBtn: {
    backgroundColor: '#6D4C41',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
  },
  smallBtnText: { color: 'white' },
  runBtn: {
    backgroundColor: '#1E88E5',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 12,
  },
  runBtnText: { color: 'white', fontWeight: '600' },
  output: {
    marginTop: 16,
    borderColor: '#eee',
    borderWidth: 1,
    borderRadius: 6,
    padding: 12,
    flex: 1,
  },
  progressTrack: {
    height: 6,
    backgroundColor: '#e0e0e0',
    borderRadius: 3,
    overflow: 'hidden',
    marginTop: 8,
  },
  progressFill: { height: 6, backgroundColor: '#1E88E5' },
  progressText: { marginTop: 4, color: '#555' },
});
