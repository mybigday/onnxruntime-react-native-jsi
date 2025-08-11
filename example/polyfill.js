import 'text-encoding-polyfill';
import { Buffer } from 'buffer';
import XRegExp from 'xregexp';
import { Float16Array } from '@petamoriken/float16';

global.Buffer = Buffer;
global.Float16Array = Float16Array;

// replace default RegExp to support unicode
const nativeRegExp = global.RegExp;
const newRegExp = (...args) => {
  global.RegExp = nativeRegExp;
  const result = XRegExp(...args);
  global.RegExp = newRegExp;
  return result;
};
global.RegExp = newRegExp;
