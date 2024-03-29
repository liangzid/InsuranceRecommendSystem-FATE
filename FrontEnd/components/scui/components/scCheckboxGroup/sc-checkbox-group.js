"use strict";

var checkboxGroupCount = 1;
Component({
  properties: {
    name: {
      type: String
    },
    direction: {
      type: String,
      value: "row"
    },
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    value: []
  },
  externalClasses: ["sc-class"],
  ready: function () {
    this.items = this._getAllCheckboxs(), this.setData({
      name: this.properties.name || "checkboxGroup" + checkboxGroupCount++
    });
    var e = !0,
        t = !1,
        a = void 0;

    try {
      for (var r, o = this.items[Symbol.iterator](); !(e = (r = o.next()).done); e = !0) {
        var c = r.value;
        c.data.checked && this.data.value.push(c.data.value);
      }
    } catch (e) {
      t = !0, a = e;
    } finally {
      try {
        !e && o.return && o.return();
      } finally {
        if (t) throw a;
      }
    }
  },
  methods: {
    _getAllCheckboxs: function () {
      return getCurrentPages()[getCurrentPages().length - 1].selectAllComponents('.scForm-' + this.data.swanIdForSystem);
    },
    _checkChange: function (e) {
      var t = this.data.value = [],
          a = !0,
          r = !1,
          o = void 0;

      try {
        for (var c, s = this.items[Symbol.iterator](); !(a = (c = s.next()).done); a = !0) {
          var i = c.value;
          i.data.checked && t.push(i.data.value);
        }
      } catch (e) {
        r = !0, o = e;
      } finally {
        try {
          !a && s.return && s.return();
        } finally {
          if (r) throw o;
        }
      }

      this.triggerEvent("change", {
        value: t
      }, {
        bubbles: !0,
        composed: !0
      });
    }
  }
});