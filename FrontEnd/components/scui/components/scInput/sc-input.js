"use strict";

var inputCount = 1;
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
    name: {
      type: String
    },
    float: {
      type: Boolean,
      value: !0
    },
    inputType: {
      type: String
    },
    errText: {
      type: String
    },
    errStatus: {
      type: Boolean
    },
    regex: {
      type: String
    },
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    value: null,
    inputTypeList: {
      phone: {
        regex: "^[1][3,4,5,7,8][0-9]{9}$",
        errText: "请输入正确的手机号"
      }
    },
    requireVerify: !1,
    verify: {
      errText: null,
      regex: null
    }
  },
  externalClasses: ["sc-class", "icon"],
  ready: function () {
    var e = this.properties,
        t = e.value,
        r = e.placeholder,
        a = e.name,
        i = e.inputType,
        s = e.regex,
        l = e.errText,
        n = this.data.inputTypeList;
    this.setData({
      value: t,
      placeholder: r,
      name: a || "input" + inputCount++
    });
    var u = n[i];
    if (i && !u) throw this.data.requireVerify = !1, new Error("no this inputType");
    (s || i && u) && (this.data.requireVerify = !0, this.data.verify.errText = l || u.errText || null, this.data.verify.regex = new RegExp(s || u.regex || null), this.setData({
      errText: this.data.verify.errText
    }));
  },
  methods: {
    _focus: function (e) {
      this.properties.disabled || (this.setData({
        focus: !0
      }), this.triggerEvent("focus", e.detail, {}));
    },
    _blur: function (e) {
      this.properties.disabled || (this.setData({
        focus: !1
      }), this.triggerEvent("blur", e, {}));
    },
    _input: function (e) {
      var t = this.properties.disabled,
          r = e.detail.value;

      if (!t) {
        if (this.data.requireVerify) {
          var a = this.data.verify,
              i = a.errText,
              s = a.regex;
          i && s && (this.setData({
            errStatus: !s.test(r)
          }), this.properties.errStatus ? (this.setData({
            value: r
          }), this.data.value = null) : this.setData({
            value: r
          }));
        } else this.setData({
          value: r
        });

        this.triggerEvent("input", e.detail, {});
      }
    },
    _confirm: function (e) {
      this.properties.disabled || this.triggerEvent("confirm", e.detail, {});
    }
  }
});