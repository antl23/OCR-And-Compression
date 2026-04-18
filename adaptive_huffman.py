from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import log2
from typing import Dict, Optional, List


ESC = "<ESC>"


@dataclass(order=True)
class HuffNode:
    weight: int
    sort_key: str
    symbol: Optional[str] = field(compare=False, default=None)
    left: Optional["HuffNode"] = field(compare=False, default=None)
    right: Optional["HuffNode"] = field(compare=False, default=None)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class AdaptiveHuffman:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.counts: Counter[str] = Counter({ESC: 1})

    @staticmethod
    def symbol_to_bits(symbol: str) -> str:
        return format(ord(symbol), "08b")

    @staticmethod
    def bits_to_symbol(bits: str) -> str:
        return chr(int(bits, 2))

    def _build_tree(self) -> HuffNode:
        heap: List[HuffNode] = []
        for symbol, weight in sorted(self.counts.items(), key=lambda kv: (kv[0] != ESC, kv[0])):
            heappush(heap, HuffNode(weight=weight, sort_key=symbol, symbol=symbol))

        if len(heap) == 1:
            only = heappop(heap)
            return HuffNode(weight=only.weight, sort_key=only.sort_key, left=only)

        while len(heap) > 1:
            a = heappop(heap)
            b = heappop(heap)
            parent = HuffNode(
                weight=a.weight + b.weight,
                sort_key=min(a.sort_key, b.sort_key),
                left=a,
                right=b,
            )
            heappush(heap, parent)
        return heap[0]

    def _build_codes(self, root: HuffNode) -> Dict[str, str]:
        codes: Dict[str, str] = {}

        def walk(node: HuffNode, prefix: str) -> None:
            if node.is_leaf:
                if node.symbol is not None:
                    codes[node.symbol] = prefix or "0"
                return
            if node.left is not None:
                walk(node.left, prefix + "0")
            if node.right is not None:
                walk(node.right, prefix + "1")

        walk(root, "")
        return codes

    def encode(self, text: str) -> str:
        self.reset()
        out: List[str] = []
        for ch in text:
            codes = self._build_codes(self._build_tree())
            if ch in self.counts:
                out.append(codes[ch])
            else:
                out.append(codes[ESC])
                out.append(self.symbol_to_bits(ch))
            self.counts[ch] += 1
        return "".join(out)

    def decode(self, bits: str) -> str:
        self.reset()
        out: List[str] = []
        i = 0
        while i < len(bits):
            root = self._build_tree()
            node = root
            while not node.is_leaf:
                if i >= len(bits):
                    raise ValueError("Unexpected end of bitstream")
                bit = bits[i]
                i += 1
                node = node.left if bit == "0" else node.right
                if node is None:
                    raise ValueError("Invalid bitstream")

            symbol = node.symbol
            if symbol == ESC:
                if i + 8 > len(bits):
                    raise ValueError("Missing raw symbol payload after ESC")
                symbol = self.bits_to_symbol(bits[i:i+8])
                i += 8

            out.append(symbol)
            self.counts[symbol] += 1
        return "".join(out)

    @staticmethod
    def pack_bits(bitstring: str) -> bytes:
        if not bitstring:
            return bytes([0])
        padding = (8 - len(bitstring) % 8) % 8
        padded = bitstring + ("0" * padding)
        header = format(padding, "08b")
        final = header + padded
        return int(final, 2).to_bytes(len(final) // 8, byteorder="big")

    @staticmethod
    def unpack_bits(blob: bytes) -> str:
        if not blob:
            return ""
        bits = "".join(format(byte, "08b") for byte in blob)
        padding = int(bits[:8], 2)
        payload = bits[8:]
        if padding:
            payload = payload[:-padding]
        return payload

    @staticmethod
    def entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        total = len(text)
        return -sum((c / total) * log2(c / total) for c in counts.values())

    @staticmethod
    def fixed_width_baseline_bits(text: str) -> int:
        return len(text) * 8

    def metrics(self, text: str, bitstring: str) -> Dict[str, float]:
        if not text:
            return {
                "original_bits": 0.0,
                "compressed_bits": 0.0,
                "compression_ratio": 1.0,
                "space_saving": 0.0,
                "entropy_bits_per_symbol": 0.0,
                "average_code_length": 0.0,
                "encoding_efficiency": 0.0,
            }
        original_bits = float(self.fixed_width_baseline_bits(text))
        compressed_bits = float(len(bitstring))
        entropy = self.entropy(text)
        average_code_length = compressed_bits / len(text)
        efficiency = (entropy / average_code_length) if average_code_length else 0.0
        return {
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": original_bits / compressed_bits if compressed_bits else 1.0,
            "space_saving": (1.0 - (compressed_bits / original_bits)) if original_bits else 0.0,
            "entropy_bits_per_symbol": entropy,
            "average_code_length": average_code_length,
            "encoding_efficiency": efficiency,
        }
