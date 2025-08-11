import { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Platform,
  PermissionsAndroid,
} from 'react-native';
import { useAI } from '../context/AIContext';
import * as DocumentPicker from '@react-native-documents/picker';
import { AudioRecorder, AudioContext } from 'react-native-audio-api';

export default function ASRScreen() {
  const { runASRFromUrl, runASRFromFloat32, isLoaded } = useAI();
  const [input, setInput] = useState('');
  const [result, setResult] = useState<string>('');
  const [isRunning, setIsRunning] = useState(false);
  const [audioUri, setAudioUri] = useState<string | undefined>(undefined);
  const [isRecording, setIsRecording] = useState(false);
  const scrollRef = useRef<ScrollView | null>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);
  const recordedChunksRef = useRef<Float32Array[]>([]);
  const recordedTotalSamplesRef = useRef<number>(0);

  useEffect(() => {
    return () => {
      recorderRef.current?.stop();
      recorderRef.current = null;
    };
  }, []);

  const onRun = useCallback(async () => {
    if (!isLoaded) return;
    setIsRunning(true);
    setResult('');
    try {
      let out;
      const recordedPCM: Float32Array | undefined = (onRun as any)._pcm;
      if (recordedPCM) {
        out = await runASRFromFloat32(recordedPCM);
        (onRun as any)._pcm = undefined;
      } else if (audioUri) {
        const audioCtx = new AudioContext();
        const resp = await fetch(audioUri);
        const arr = await resp.arrayBuffer();
        const decoded = await audioCtx.decodeAudioData(arr);
        const ch0 = decoded.getChannelData
          ? decoded.getChannelData(0)
          : new Float32Array();
        out = await runASRFromFloat32(ch0);
      } else {
        out = await runASRFromUrl(input);
      }
      setResult(out?.text ?? '');
    } catch (e: any) {
      setResult(String(e?.message ?? e));
    } finally {
      setIsRunning(false);
      requestAnimationFrame(() =>
        scrollRef.current?.scrollToEnd({ animated: true })
      );
    }
  }, [isLoaded, input, audioUri, runASRFromFloat32, runASRFromUrl]);

  const pickAudio = useCallback(async () => {
    try {
      const res = await DocumentPicker.pick({
        type: [DocumentPicker.types.audio],
      });
      setAudioUri(res[0].uri);
    } catch (e: any) {
      const isCanceled =
        e?.code === 'DOCUMENT_PICKER_CANCELED' || e?.name === 'CanceledError';
      if (!isCanceled) {
        setResult(String(e?.message ?? e));
      }
    }
  }, []);

  const startRecord = useCallback(async () => {
    try {
      if (Platform.OS === 'android') {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO
        );
        if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
          setResult('Microphone permission denied');
          return;
        }
      }
      if (!recorderRef.current) {
        recorderRef.current = new AudioRecorder({
          sampleRate: 16000,
          bufferLengthInSamples: 16000,
        });
      }
      // Re-bind handler each start to ensure fresh closure
      recorderRef.current.onAudioReady((event: any) => {
        const pcm = extractPcmFromEvent(event);
        if (pcm && pcm.length > 0) {
          // Copy to avoid underlying buffer reuse
          const copy = new Float32Array(pcm);
          recordedChunksRef.current.push(copy);
          recordedTotalSamplesRef.current += copy.length;
        }
      });
      recordedChunksRef.current = [];
      recordedTotalSamplesRef.current = 0;
      await recorderRef.current.start();
      setIsRecording(true);
      setAudioUri(undefined);
      setResult('Recording...');
    } catch (e: any) {
      setResult(String(e?.message ?? e));
    }
  }, []);

  const stopRecord = useCallback(async () => {
    try {
      if (!recorderRef.current) return;
      await recorderRef.current.stop();
      setIsRecording(false);
      const total = recordedTotalSamplesRef.current;
      const chunks = recordedChunksRef.current;
      if (total > 0 && chunks.length > 0) {
        const out = new Float32Array(total);
        let offset = 0;
        for (const c of chunks) {
          out.set(c, offset);
          offset += c.length;
        }
        (onRun as any)._pcm = out;
        setResult(`Recorded ${total} samples @ 16000 Hz`);
      } else {
        setResult('Recorded 0 samples. Please try again.');
      }
    } catch (e: any) {
      setResult(String(e?.message ?? e));
    }
  }, [onRun]);

  function extractPcmFromEvent(event: any): Float32Array | null {
    const numFrames: number | undefined = event?.numFrames;
    const fromBuffer = (buf: any): Float32Array | null => {
      try {
        if (!buf) return null;
        if (typeof buf.getChannelData === 'function') {
          const ch0: Float32Array = buf.getChannelData(0);
          if (numFrames && numFrames > 0 && numFrames <= ch0.length) {
            return new Float32Array(ch0.subarray(0, numFrames));
          }
          return new Float32Array(ch0);
        }
        if (buf instanceof Float32Array) return buf;
        if (Array.isArray(buf)) return new Float32Array(buf as number[]);
      } catch {}
      return null;
    };
    const c1 = fromBuffer(event?.buffer);
    if (c1 && c1.length) return c1;
    const c2 = fromBuffer(event?.pcm ?? event?.data);
    return c2 && c2.length ? c2 : null;
  }

  return (
    <View style={styles.container}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>ASR</Text>
      </View>
      <TextInput
        style={styles.input}
        value={input}
        onChangeText={setInput}
        placeholder={'Enter audio URL (wav/mp3) or pick/record below'}
        autoCapitalize="none"
        autoCorrect={false}
      />
      <View style={styles.row}>
        <TouchableOpacity style={styles.smallBtn} onPress={pickAudio}>
          <Text style={styles.smallBtnText}>Pick audio file</Text>
        </TouchableOpacity>
        <View style={{ width: 8 }} />
        {!isRecording ? (
          <TouchableOpacity
            style={[styles.smallBtn, { backgroundColor: '#6D4C41' }]}
            onPress={startRecord}
          >
            <Text style={styles.smallBtnText}>Start recording</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[styles.smallBtn, { backgroundColor: '#B71C1C' }]}
            onPress={stopRecord}
          >
            <Text style={styles.smallBtnText}>Stop</Text>
          </TouchableOpacity>
        )}
      </View>
      {audioUri && (
        <Text style={{ marginTop: 6 }} numberOfLines={1}>
          Selected: {audioUri}
        </Text>
      )}
      <TouchableOpacity
        style={styles.runBtn}
        onPress={onRun}
        disabled={!isLoaded || isRunning}
      >
        <Text style={styles.runBtnText}>
          {isRunning ? 'Running...' : 'Transcribe'}
        </Text>
      </TouchableOpacity>
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
});
