"use strict";

var TimePicker = function () {
  var t = function (t) {
    if (this.id = t || null, this.id) {
      var e = getCurrentPages(),
          i = e[e.length - 1];
      if (this.timePicker = i.selectComponent(this.id), this.timePicker) return this;
      throw new Error("no this id of sc-time-picker");
    }
  };

  return t.prototype.open = function (t) {
    return this.timePicker._open(t), this;
  }, t.prototype.close = function () {
    return this.timePicker._close(), this;
  }, function (e) {
    return new t(e);
  };
}();

module.exports = TimePicker;