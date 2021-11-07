"use strict";

var scRippleBehaviors = require("../sc-ripple-behaviors/sc-ripple-behaviors");

Component({
  behaviors: [scRippleBehaviors],
  properties: {
    reportSubmit: {
      type: Boolean,
      value: !1
    },
    submitText: {
      type: String
    },
    showSubmit: {
      type: Boolean,
      value: !0
    },
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    btnSelector: ".submit-btn-class"
  },
  externalClasses: ["sc-class", "sc-button-class"],
  ready: function () {
    this.formControllers = this._getAllControl();
  },
  methods: {
    _getAllControl: function () {
      return {
        checkboxGroups: getCurrentPages()[getCurrentPages().length - 1].selectAllComponents('.scRadioGroup-' + this.data.swanIdForSystem),
        inputs: getCurrentPages()[getCurrentPages().length - 1].selectAllComponents('.scRadioGroup-' + this.data.swanIdForSystem),
        textareas: getCurrentPages()[getCurrentPages().length - 1].selectAllComponents('.scRadioGroup-' + this.data.swanIdForSystem),
        switchs: getCurrentPages()[getCurrentPages().length - 1].selectAllComponents('.scRadioGroup-' + this.data.swanIdForSystem),
        radioGroups: getCurrentPages()[getCurrentPages().length - 1].selectAllComponents('.scRadioGroup-' + this.data.swanIdForSystem)
      };
    },
    _formSubmit: function (t) {
      var e = this.formControllers,
          s = {
        formId: t.detail.formId
      };

      for (var o in e) e.hasOwnProperty(o) && e[o].length > 0 && e[o].forEach(function (t) {
        s[t.data.name] = t.data.value || null;
      });

      this.triggerEvent("submit", {
        value: s
      });
    },
    _tap: function (t) {
      this._addRipple_(t);
    },
    _longPress: function (t) {
      this._longPress_(t);
    },
    _rippleAnimationEnd: function () {
      this._rippleAnimationEnd_();
    },
    _touchEnd: function () {
      this._touchEnd_();
    }
  }
});