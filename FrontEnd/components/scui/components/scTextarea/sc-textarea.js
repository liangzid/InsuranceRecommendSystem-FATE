"use strict";

var textareaCount = 1;
Component({
  properties: {
    label: {
      type: String
    },
    value: {
      type: String
    },
    type: {
      type: String,
      value: "text"
    },
    password: {
      type: Boolean,
      value: !1
    },
    placeholder: {
      type: String
    },
    placeholderStyle: {
      type: String
    },
    placeholderClass: {
      type: String,
      value: "input-placeholder"
    },
    disabled: {
      type: Boolean,
      value: !1
    },
    maxlength: {
      type: Number,
      value: 140
    },
    cursorSpacing: {
      type: Number,
      value: 0
    },
    focus: {
      type: Boolean,
      value: !1
    },
    confirmType: {
      type: String,
      value: "done"
    },
    confirmHold: {
      type: Boolean,
      value: !1
    },
    cursor: {
      type: Number
    },
    selectionStart: {
      type: Number,
      value: -1
    },
    selectionEnd: {
      type: Number,
      value: -1
    },
    adjustPosition: {
      type: Boolean,
      value: !0
    },
    showConfirmBar: {
      type: Boolean,
      value: !0
    },
    fixed: {
      type: Boolean,
      value: !1
    },
    autoHeight: {
      type: Boolean,
      value: !1
    },
    height: {
      type: String,
      value: "64rpx"
    },
    name: {
      type: String
    },
    float: {
      type: Boolean,
      value: !0
    },
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    value: null
  },
  externalClasses: ["sc-class"],
  ready: function () {
    this.setData({
      value: this.properties.value,
      placeholder: this.properties.label ? null : this.properties.placeholder,
      name: this.properties.name || "textarea" + textareaCount++
    });
  },
  methods: {
    _focus: function (e) {
      this.properties.disabled || (this.setData({
        focus: !0
      }), this.triggerEvent("focus", e, {}));
    },
    _blur: function (e) {
      this.properties.disabled || (this.setData({
        focus: !1
      }), this.triggerEvent("blur", e, {}));
    },
    _input: function (e) {
      this.properties.disabled || (this.setData({
        value: e.detail.value
      }), this.triggerEvent("input", e.detail, {}));
    },
    _confirm: function (e) {
      this.properties.disabled || this.triggerEvent("confirm", e.detail, {});
    },
    _linechange: function (e) {
      this.properties.disabled || this.triggerEvent("linechange", e.detail, {});
    }
  }
});