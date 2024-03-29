"use strict";

var config = require("./config/config"),
    Dialog = require("./components/scDialog/sc-dialog-o"),
    SnackBar = require("./components/scSnackbar/sc-snackbar-o"),
    DatePicker = require("./components/scDatePicker/sc-date-picker-o"),
    TimePicker = require("./components/scTimePicker/sc-time-picker-o"),
    scui = {};

Object.defineProperty(scui, "version", {
  configurable: !1,
  writable: !1,
  enumerable: !1,
  value: config.version
}), module.exports = function (e) {
  return e = e || {}, e.Dialog = Dialog, e.SnackBar = SnackBar, e.DatePicker = DatePicker, e.TimePicker = TimePicker, e;
}(scui);