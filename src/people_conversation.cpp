#include "people_view.hpp"
#include <QPainter>
#include <algorithm>
#include <cmath>
#include <map>

namespace sona {
namespace {
QColor speakerColor(int id) {
    return QColor::fromHsvF(std::fmod(0.47 + id * 0.19, 1.0), 0.35, 0.85);
}
QString speakerName(int id) {
    return QString("Voice %1").arg(id + 1);
}
QString seconds(double value) {
    return QString::number(value, 'f', 1) + " s";
}
struct LinkVertex {
    float x, y, z, nx = 0, ny = 1, nz = 0;
};
} // namespace
void PeopleView::enableConversation() {
    conversation_ = true;
    setObjectName("conversationView");
    setAccessibleName(
        "Anonymous conversation in 3D. Drag to orbit, wheel to zoom; click timeline to seek a recording.");
    resetCamera();
}
void PeopleView::setConversationData(const std::vector<SpeechSegment> &segments, double analyzedThrough,
                                     const std::vector<TimeSpan> &coverage) {
    conversationSegments_ = segments;
    conversationCoverage_ = coverage;
    analyzedThrough_ = analyzedThrough;
    setConversationTime(conversationTime_, seekable_);
}
void PeopleView::setConversationTime(double seconds, bool seekable) {
    conversationTime_ = std::max(0.0, seconds);
    seekable_ = seekable;
    conversationSummary_ = summarizeConversation(
        conversationSegments_, std::min(conversationTime_, analyzedThrough_), conversationCoverage_);
    if (!seekable_)
        for (auto &speaker : conversationSummary_.speakers)
            speaker.active = speaker.active && conversationTime_ - analyzedThrough_ <= 2.5;
    update();
}
void PeopleView::setConversationStatus(const QString &status) {
    conversationStatus_ = status;
    update();
}
int PeopleView::timelineHeight() const {
    return 72 + int(std::min(std::size_t(8), conversationSummary_.speakers.size())) * 23;
}
std::vector<QVector3D> PeopleView::conversationPositions() const {
    const int count = int(std::min(std::size_t(8), conversationSummary_.speakers.size()));
    std::vector<QVector3D> positions;
    for (int i = 0; i < count; ++i) {
        const float x = float(i - (count - 1) * 0.5) * (count <= 4 ? 1.12f : 0.82f);
        positions.emplace_back(x, 0, std::abs(x) * 0.22f);
    }
    return positions;
}
void PeopleView::drawConnections(const QMatrix4x4 &mvp) {
    const auto positions = conversationPositions();
    std::map<int, QVector3D> location;
    for (std::size_t i = 0; i < positions.size(); ++i)
        location[conversationSummary_.speakers[i].id] = positions[i];
    std::map<std::pair<int, int>, double> latest;
    for (const auto &e : conversationSummary_.exchanges)
        if (conversationTime_ - e.at < 30 && location.count(e.from) && location.count(e.to))
            latest[{e.from, e.to}] = e.at;
    program_.setUniformValue("mvp", mvp);
    program_.setUniformValue("pass", 5);
    program_.setUniformValue("activity", 0.0f);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);
    linksVao_.bind();
    links_.bind();
    for (const auto &[pair, at] : latest) {
        const auto a = location[pair.first], b = location[pair.second];
        std::vector<LinkVertex> vertices;
        const auto point = [a, b](float t) {
            auto p = a * (1 - t) + b * t;
            p.setY(0.025f + std::sin(t * float(pi)) * 0.24f);
            p.setZ(p.z() + std::sin(t * float(pi)) * 0.25f);
            return p;
        };
        for (int i = 0; i < 40; ++i) {
            const auto p = point(i / 40.0f), q = point((i + 1) / 40.0f);
            vertices.push_back({p.x(), p.y(), p.z()});
            vertices.push_back({q.x(), q.y(), q.z()});
        }
        const float progress = float(std::fmod(std::max(0.0, conversationTime_ - at) * 0.45, 1.0));
        const auto marker = point(progress), back = point(std::max(0.0f, progress - 0.055f));
        for (float side : {-1.0f, 1.0f}) {
            const auto wing = back + QVector3D(0, 0.045f * side, 0);
            vertices.push_back({marker.x(), marker.y(), marker.z()});
            vertices.push_back({wing.x(), wing.y(), wing.z()});
        }
        links_.allocate(vertices.data(), int(vertices.size() * sizeof(LinkVertex)));
        const auto color = speakerColor(pair.second);
        const float strength = float(0.5 + 0.5 * std::exp(-(conversationTime_ - at) / 4));
        program_.setUniformValue("tint",
                                 QVector3D(float(color.redF()) * strength, float(color.greenF()) * strength,
                                           float(color.blueF()) * strength));
        glDrawArrays(GL_LINES, 0, GLsizei(vertices.size()));
    }
    linksVao_.release();
    glDepthMask(GL_TRUE);
}
void PeopleView::drawConversationLabels(const QMatrix4x4 &mvp, int sceneHeight) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    auto font = painter.font();
    font.setPixelSize(12);
    painter.setFont(font);
    painter.setPen(QColor("#c1cdd2"));
    painter.drawText(20, 28, "Anonymous voices · Generic proportions");
    painter.setPen(QColor("#81949e"));
    painter.drawText(20, 49,
                     painter.fontMetrics().elidedText(conversationStatus_, Qt::ElideRight, width() - 40));
    painter.setPen(QColor("#81949e"));
    painter.drawText(QRect(width() - 240, 12, 220, 20), Qt::AlignRight,
                     (seekable_ ? "Timeline " : "Analyzed through ") +
                         seconds(conversationSummary_.duration));
    const auto positions = conversationPositions();
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto &person = conversationSummary_.speakers[i];
        const auto clip = mvp * QVector4D(positions[i] + QVector3D(0, 1.86f, 0), 1);
        if (clip.w() <= 0)
            continue;
        const auto projected = clip.toVector3DAffine();
        const int x = int((projected.x() + 1) * width() / 2), y = int((1 - projected.y()) * sceneHeight / 2);
        painter.setPen(speakerColor(person.id));
        painter.drawText(QRect(x - 85, y - 14, 170, 20), Qt::AlignCenter,
                         speakerName(person.id) +
                             (person.active ? (seekable_ ? " · speaking" : " · detected") : ""));
    }
    const int top = height() - timelineHeight();
    painter.fillRect(QRect(0, top, width(), timelineHeight()), QColor("#151b21"));
    painter.setPen(QColor("#34414a"));
    painter.drawLine(20, top, width() - 20, top);
    const auto &s = conversationSummary_;
    painter.setPen(QColor("#aabac2"));
    painter.drawText(20, top + 23,
                     QString("%1 voices    %2 speech    %3 overlap    %4 silence")
                         .arg(s.speakers.size())
                         .arg(seconds(s.speech), seconds(s.overlap), seconds(s.silence)));
    if (s.speakers.size() > positions.size())
        painter.drawText(width() - 190, top + 23, "Showing first eight voices");
    else if (s.unobserved > 0.1)
        painter.drawText(width() - 190, top + 23, seconds(s.unobserved) + " not analyzed");
    if (positions.empty()) {
        painter.setPen(QColor("#748892"));
        painter.drawText(20, top + 48, "Figures appear as speech is assigned to separate voices.");
        return;
    }
    const double span = std::min(60.0, analyzedThrough_);
    const double begin = std::clamp(conversationTime_ - 45, 0.0, std::max(0.0, analyzedThrough_ - span));
    const int left = 100, plotWidth = std::max(1, width() - 235);
    const auto xAt = [&](double at) { return left + plotWidth * (at - begin) / std::max(1.0, span); };
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto &person = s.speakers[i];
        const int y = top + 39 + int(i) * 23;
        const auto color = speakerColor(person.id);
        painter.setPen(color);
        painter.drawText(20, y + 12, speakerName(person.id));
        painter.fillRect(QRect(left, y, plotWidth, 14), QColor("#202a32"));
        painter.setPen(Qt::NoPen);
        for (const auto &segment : conversationSegments_)
            if (segment.speaker == person.id && segment.end > begin && segment.start < begin + span) {
                const double x = xAt(std::max(begin, segment.start));
                painter.setBrush(segment.start <= conversationTime_ ? color : color.darker(220));
                painter.drawRoundedRect(
                    QRectF(x, y + 2, std::max(1.0, xAt(std::min(begin + span, segment.end)) - x), 10), 2, 2);
            }
        painter.setPen(QColor("#aabac2"));
        painter.drawText(left + plotWidth + 12, y + 12,
                         QString("%1% · %2").arg(person.share * 100, 0, 'f', 0).arg(seconds(person.seconds)));
    }
    const int bottom = top + 39 + int(positions.size()) * 23;
    if (conversationTime_ >= begin && conversationTime_ <= begin + span) {
        painter.setPen(QColor("#d5e2e7"));
        painter.drawLine(QPointF(xAt(conversationTime_), top + 35),
                         QPointF(xAt(conversationTime_), bottom - 6));
    }
    painter.setPen(QColor("#748892"));
    painter.drawText(left, bottom + 12, seconds(begin));
    painter.drawText(left + plotWidth - 65, bottom + 12, seconds(begin + span));
    std::vector<double> gaps;
    for (const auto &exchange : s.exchanges)
        if (exchange.hasResponseGap)
            gaps.push_back(exchange.gap);
    QString exchangeText = QString("%1 turn exchanges").arg(s.exchanges.size());
    if (!gaps.empty()) {
        std::sort(gaps.begin(), gaps.end());
        const auto mid = gaps.size() / 2;
        const double median = gaps.size() % 2 ? gaps[mid] : (gaps[mid - 1] + gaps[mid]) * 0.5;
        exchangeText += " · median response gap " + seconds(median);
    }
    painter.drawText(width() / 2 - 150, bottom + 12, exchangeText);
}
} // namespace sona
