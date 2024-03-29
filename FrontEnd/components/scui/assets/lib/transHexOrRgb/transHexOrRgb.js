"use strict";

var reg = /^#([0-9a-fA-f]{3}|[0-9a-fA-f]{6})$/,
    colorHex = function (r) {
  var e = r;

  if (/^(rgb|RGB)/.test(e)) {
    for (var t = e.replace(/(?:\(|\)|rgb|RGB)*/g, "").split(","), o = "#", l = 0; l < t.length; l++) {
      var n = Number(t[l]).toString(16);
      "0" === n && (n += n), o += n;
    }

    return 7 !== o.length && (o = e), o;
  }

  if (!reg.test(e)) return e;
  var a = e.replace(/#/, "").split("");
  if (6 === a.length) return e;

  if (3 === a.length) {
    for (var g = "#", i = 0; i < a.length; i += 1) g += a[i] + a[i];

    return g;
  }
},
    colorRgb = function (r, e) {
  var t = r.toLowerCase();

  if (t && reg.test(t.trim())) {
    if (4 === t.length) {
      for (var o = "#", l = 1; l < 4; l += 1) o += t.slice(l, l + 1).concat(t.slice(l, l + 1));

      t = o;
    }

    for (var n = [], a = 1; a < 7; a += 2) n.push(parseInt("0x" + t.slice(a, a + 2)));

    return e && n.push(e), "rgba(" + n.join(",") + ")";
  }

  return t;
};

module.exports = {
  colorHex: colorHex,
  colorRgb: colorRgb
};