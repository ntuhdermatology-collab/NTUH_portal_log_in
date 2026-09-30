import unittest

from selenium.common.exceptions import NoAlertPresentException

import portal_log_in


class _FakeAlert:
    def __init__(self, text):
        self.text = text
        self.accepted = False

    def accept(self):
        self.accepted = True


class _FakeSwitchTo:
    def __init__(self, alert=None):
        self._alert = alert

    @property
    def alert(self):
        if self._alert is None:
            raise NoAlertPresentException()
        return self._alert


class _FakeElement:
    def __init__(self, text, *, displayed=True):
        self.text = text
        self._displayed = displayed

    def is_displayed(self):
        return self._displayed


class _FakeDriver:
    def __init__(self, *, alert=None, inline_message=""):
        self.switch_to = _FakeSwitchTo(alert)
        self._inline_message = inline_message

    def find_elements(self, by, value):
        if value != "lblMessage" or not self._inline_message:
            return []
        return [_FakeElement(self._inline_message)]


class PortalLoginProtectionTests(unittest.TestCase):
    def test_success_wait_tolerates_null_element_from_selenium(self):
        class Driver:
            @staticmethod
            def find_elements(by, value):
                return [None]

        predicate = portal_log_in._visible_element_or_false(("id", "success"))

        self.assertFalse(predicate(Driver()))

    def test_success_wait_returns_visible_element(self):
        visible_element = _FakeElement("", displayed=True)

        class Driver:
            @staticmethod
            def find_elements(by, value):
                return [None, visible_element]

        predicate = portal_log_in._visible_element_or_false(("id", "success"))

        self.assertIs(predicate(Driver()), visible_element)

    def test_live_portal_credential_message_is_not_captcha(self):
        message = (
            "帳號密碼錯誤，請重新確認與輸入，"
            "若還有問題，請洽資訊服務台#261120"
        )
        self.assertEqual(
            portal_log_in._classify_login_failure(message),
            "credentials",
        )

    def test_only_explicit_captcha_message_is_retryable(self):
        self.assertEqual(
            portal_log_in._classify_login_failure("驗證碼輸入錯誤"),
            "captcha",
        )
        self.assertEqual(
            portal_log_in._classify_login_failure("系統暫時無法處理"),
            "unknown",
        )
        self.assertEqual(portal_log_in._classify_login_failure(""), "unknown")

    def test_alert_is_read_and_accepted(self):
        alert = _FakeAlert("密碼錯誤")
        driver = _FakeDriver(alert=alert)

        message = portal_log_in._read_login_failure_message(driver)

        self.assertEqual(message, "密碼錯誤")
        self.assertTrue(alert.accepted)

    def test_inline_message_is_used_after_popup_was_dismissed(self):
        live_message = "帳號密碼錯誤，請重新確認與輸入"
        driver = _FakeDriver(inline_message=live_message)

        message = portal_log_in._read_login_failure_message(driver)

        self.assertEqual(message, live_message)

    def test_duplicate_alert_and_inline_text_is_returned_once(self):
        live_message = "帳號密碼錯誤"
        alert = _FakeAlert(live_message)
        driver = _FakeDriver(alert=alert, inline_message=live_message)

        message = portal_log_in._read_login_failure_message(
            driver,
            fallback_message=live_message,
        )

        self.assertEqual(message, live_message)


if __name__ == "__main__":
    unittest.main()
